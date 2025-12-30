use anyhow::{Result, anyhow};
use collections::BTreeMap;
use futures::{FutureExt, Stream, StreamExt, future, future::BoxFuture, stream::BoxStream};
use futures::stream::TryStreamExt;
use gpui::{AnyView, App, AsyncApp, Context, Entity, FocusHandle, FontWeight, SharedString, Task, Window, actions};
use http_client::HttpClient;

use language_model::{
    ApiKeyState, AuthenticateError, ConfigurationViewTargetAgent, EnvVar, IconOrSvg, LanguageModel,
    LanguageModelCacheConfiguration, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, RateLimiter, Role, StopReason, env_var,
};
use menu;
use serde::{Deserialize, Serialize};
use serde_json;
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use thiserror::Error;
use ui::{ButtonLink, ConfiguredApiCard, ElevationIndex, Icon, IconName, Label, LabelSize, List, ListBulletItem, prelude::*, Color, Tooltip};
use crate::AllLanguageModelSettings;
use ui::div;
use ui_input::InputField;
use ui::v_flex;
use util::ResultExt;

pub use settings::ZhipuAvailableModel as AvailableModel;

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("zhipu");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Zhipu AI");

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ZhipuSettings {
    pub api_url: String,
    /// Extend Zed's list of Zhipu AI models.
    pub available_models: Vec<AvailableModel>,
}

pub struct ZhipuLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

const API_KEY_ENV_VAR_NAME: &str = "ZHIPU_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

pub struct State {
    api_key_state: ApiKeyState,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let api_url = ZhipuLanguageModelProvider::api_url(cx);
        self.api_key_state
            .store(api_url, api_key, |this| &mut this.api_key_state, cx)
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let api_url = ZhipuLanguageModelProvider::api_url(cx);
        self.api_key_state
            .load_if_needed(api_url, |this| &mut this.api_key_state, cx)
    }
}

impl ZhipuLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let api_url = Self::api_url(cx);
                this.api_key_state
                    .handle_url_change(api_url, |this| &mut this.api_key_state, cx);
                cx.notify();
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
            }
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: ZhipuModel) -> Arc<dyn LanguageModel> {
        Arc::new(ZhipuModelInstance {
            id: LanguageModelId::from(model.name().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn settings(cx: &App) -> &ZhipuSettings {
        &crate::AllLanguageModelSettings::get_global(cx).zhipu
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            "https://open.bigmodel.cn/api/paas/v4/chat/completions".into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }
}

impl LanguageModelProviderState for ZhipuLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for ZhipuLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::Ai)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(ZhipuModel::default()))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(ZhipuModel::default_fast()))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Add base models
        for model in ZhipuModel::all_models() {
            if !matches!(model, ZhipuModel::Custom { .. }) {
                models.insert(model.name().to_string(), model);
            }
        }

        // Override with available models from settings
        for model in &ZhipuLanguageModelProvider::settings(cx).available_models {
            models.insert(
                model.name.clone(),
                ZhipuModel::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens as usize,
                    max_output_tokens: None,
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        target_agent: ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), target_agent, window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(None, cx))
    }
}

#[derive(Clone)]
pub enum ZhipuModel {
    GLM4,
    GLM4Flash,
    GLM4Plus,
    GLM4AllTools,
    GLM47,  // GLM4.7 model
    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: usize,
        max_output_tokens: Option<u64>,
    },
}

impl ZhipuModel {
    pub fn all_models() -> Vec<Self> {
        vec![
            Self::GLM4,
            Self::GLM4Flash,
            Self::GLM4Plus,
            Self::GLM4AllTools,
            Self::GLM47,
        ]
    }

    pub fn default() -> Self {
        Self::GLM4
    }

    pub fn default_fast() -> Self {
        Self::GLM47  // Use GLM4.7 as the default fast model
    }

    pub fn name(&self) -> &str {
        match self {
            Self::GLM4 => "glm-4",
            Self::GLM4Flash => "glm-4-flash",
            Self::GLM4Plus => "glm-4-plus",
            Self::GLM4AllTools => "glm-4-alltools",
            Self::GLM47 => "glm-4-7",
            Self::Custom { name, .. } => name,
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::GLM4 => "GLM-4",
            Self::GLM4Flash => "GLM-4-Flash",
            Self::GLM4Plus => "GLM-4-Plus",
            Self::GLM4AllTools => "GLM-4-AllTools",
            Self::GLM47 => "GLM-4-7",
            Self::Custom {
                display_name,
                name,
                ..
            } => display_name.as_deref().unwrap_or(name),
        }
    }

    pub fn max_token_count(&self) -> u64 {
        match self {
            Self::GLM4 => 128_000,
            Self::GLM4Flash => 128_000,
            Self::GLM4Plus => 128_000,
            Self::GLM4AllTools => 128_000,
            Self::GLM47 => 128_000,
            Self::Custom { max_tokens, .. } => *max_tokens as u64,
        }
    }

    pub fn max_output_tokens(&self) -> u64 {
        match self {
            Self::GLM4 => 4096,
            Self::GLM4Flash => 4096,
            Self::GLM4Plus => 4096,
            Self::GLM4AllTools => 4096,
            Self::GLM47 => 4096,
            Self::Custom { .. } => 4096, // Use default value for custom models
        }
    }
}

pub struct ZhipuModelInstance {
    id: LanguageModelId,
    model: ZhipuModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl ZhipuModelInstance {
    fn stream_completion(
        &self,
        request: ZhipuRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<ZhipuResponseEvent, ZhipuError>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();

        let Ok((api_key, api_url)) = self.state.read_with(cx, |state, cx| {
            let api_url = ZhipuLanguageModelProvider::api_url(cx);
            (state.api_key_state.key(&api_url), api_url)
        }) else {
            return future::ready(Err(anyhow!("App state dropped").into())).boxed();
        };

        async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            };
            let api_key = api_key.clone();
            let api_url = api_url.clone();
            let http_client = http_client.clone();
            
            let stream = stream_completion(
                http_client.as_ref(),
                &api_url,
                &api_key,
                request,
            );
            stream.await.map_err(Into::into).map(|stream| stream.boxed())
        }
        .boxed()
    }
}

// Zhipu API request/response structures
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ZhipuRequest {
    pub model: String,
    pub messages: Vec<ZhipuMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: bool,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ZhipuMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub enum ZhipuResponseEvent {
    Content(String),
    Done,
}

#[derive(Debug, Deserialize)]
struct ZhipuResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ZhipuChoice>,
    usage: Option<ZhipuUsage>,
}

#[derive(Debug, Deserialize)]
struct ZhipuChoice {
    index: u32,
    delta: ZhipuDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZhipuDelta {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZhipuUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Error)]
pub enum ZhipuError {
    #[error("API error: {message}")]
    ApiError { message: String },
    #[error("HTTP error: {status}")]
    HttpError { status: u16 },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<http_client::http::Error> for ZhipuError {
    fn from(error: http_client::http::Error) -> Self {
        ZhipuError::Other(anyhow::Error::from(error))
    }
}

// Convert Zed LanguageModelRequest to Zhipu request
fn into_zhipu(
    request: LanguageModelRequest,
    model: String,
    max_output_tokens: u64,
) -> ZhipuRequest {
    let mut messages = Vec::new();
    
    for message in request.messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };
        
        // Convert message content to string
        let content = message.string_contents();
        
        messages.push(ZhipuMessage {
            role: role.to_string(),
            content,
        });
    }

    ZhipuRequest {
        model,
        messages,
        temperature: request.temperature,
        top_p: None, // Zhipu doesn't typically use top_p
        max_tokens: Some(max_output_tokens as u32),
        stream: true,
    }
}

// Stream completion function for Zhipu API
async fn stream_completion(
    http_client: &dyn HttpClient,
    api_url: &str,
    api_key: &str,
    request: ZhipuRequest,
) -> Result<impl Stream<Item = Result<ZhipuResponseEvent, ZhipuError>> + use<>, ZhipuError> {
    use http_body_util::BodyExt;
    use http_client::http::{HeaderValue, Method};
    use http_client::AsyncBody;
    
    let json_request = serde_json::to_string(&request)
        .map_err(|e| ZhipuError::Other(anyhow::Error::from(e)))?;

    let http_request = http_client::Request::builder()
        .method(Method::POST)
        .uri(api_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .body(AsyncBody::from(json_request));

    let response = http_client.send(http_request?).await.map_err(|e| ZhipuError::Other(e.into()))?;

    if !response.status().is_success() {
        return Err(ZhipuError::HttpError {
            status: response.status().as_u16(),
        });
    }

    use futures::io::BufReader;
    use futures::AsyncBufReadExt;
    
    // Process the response stream
    let reader = BufReader::new(response.into_body());
    let lines = reader.lines();
    
    // Process the stream to extract content
    let processed_stream = lines
        .filter_map(|line| async move {
            match line {
                Ok(line) => {
                    // Parse SSE format
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if json_str.trim() == "[DONE]" {
                            return Some(Ok(ZhipuResponseEvent::Done));
                        }
                        
                        // Try to parse the response
                        if let Ok(response_chunk) = serde_json::from_str::<ZhipuResponse>(json_str) {
                            if let Some(choice) = response_chunk.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    return Some(Ok(ZhipuResponseEvent::Content(content.clone())));
                                } else if choice.finish_reason.is_some() {
                                    return Some(Ok(ZhipuResponseEvent::Done));
                                }
                            }
                        }
                    }
                    None
                }
                Err(e) => Some(Err(ZhipuError::Other(anyhow::anyhow!("Failed to read line: {}", e)))),
            }
        });
    
    Ok(processed_stream)
}

impl LanguageModel for ZhipuModelInstance {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        matches!(self.model, ZhipuModel::GLM4AllTools | ZhipuModel::Custom { .. })
    }

    fn supports_images(&self) -> bool {
        false // Basic Zhipu models don't support images in this implementation
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto | LanguageModelToolChoice::Any | LanguageModelToolChoice::None => {
                self.supports_tools()
            }
        }
    }

    fn telemetry_id(&self) -> String {
        format!("zhipu/{}", self.model.name())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(self.model.max_output_tokens())
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        // Using a fallback token counting method for Zhipu
        cx.background_spawn(async move {
            // This is a simplified token counting implementation
            // In a real implementation, we would call Zhipu's token counting API
            let messages = request.messages;
            let mut token_count = 0;
            
            for message in messages {
                let content = message.string_contents();
                // Rough estimation: 1 token ~ 4 characters for Chinese/English mixed text
                token_count += (content.len() / 4) as u64;
            }
            
            Ok(token_count)
        })
        .boxed()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let request = into_zhipu(
            request,
            self.model.name().to_string(),
            self.model.max_output_tokens(),
        );
        let request = self.stream_completion(request, cx);
        let future = self.request_limiter.stream(async move {
            let response = request.await?;
            Ok(ZhipuEventMapper::new().map_stream(response))
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

pub struct ZhipuEventMapper {
    // Add any state needed for mapping events
}

impl ZhipuEventMapper {
    pub fn new() -> Self {
        Self {}
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<ZhipuResponseEvent, ZhipuError>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(error.into())],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: ZhipuResponseEvent,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        match event {
            ZhipuResponseEvent::Content(text) => {
                vec![Ok(LanguageModelCompletionEvent::Text(text))]
            }
            ZhipuResponseEvent::Done => {
                vec![Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn))]
            }
        }
    }
}

impl From<ZhipuError> for LanguageModelCompletionError {
    fn from(error: ZhipuError) -> Self {
        use http_client::http::StatusCode;
        match error {
            ZhipuError::ApiError { message } => LanguageModelCompletionError::AuthenticationError {
                provider: PROVIDER_NAME,
                message,
            },
            ZhipuError::HttpError { status } => {
                LanguageModelCompletionError::from_http_status(
                    PROVIDER_NAME,
                    StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    format!("HTTP error {}", status),
                    None,
                )
            }
            ZhipuError::Other(error) => LanguageModelCompletionError::Other(error),
        }
    }
}

struct ConfigurationView {
    api_key_editor: Entity<InputField>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
    target_agent: ConfigurationViewTargetAgent,
}

impl ConfigurationView {
    const PLACEHOLDER_TEXT: &'static str = "your-zhipu-api-key-here";

    fn new(
        state: Entity<State>,
        target_agent: ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn({
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor: cx.new(|cx| InputField::new(window, cx, Self::PLACEHOLDER_TEXT)),
            state,
            load_credentials_task,
            target_agent,
        }
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if api_key.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))?
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))?
                .await
        })
        .detach_and_log_err(cx);
    }

    fn should_render_editor(&self, cx: &mut Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let env_var_set = self.state.read(cx).api_key_state.is_from_env_var();
        let configured_card_label = if env_var_set {
            format!("API key set in {API_KEY_ENV_VAR_NAME} environment variable")
        } else {
            let api_url = ZhipuLanguageModelProvider::api_url(cx);
            format!("API key configured for {}", api_url)
        };

        if self.load_credentials_task.is_some() {
            div()
                .child(Label::new("Loading credentials..."))
                .into_any_element()
        } else if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new(format!("To use {}, you need to add an API key. Follow these steps:", match &self.target_agent {
                    ConfigurationViewTargetAgent::ZedAgent => "Zed's agent with Zhipu AI".into(),
                    ConfigurationViewTargetAgent::Other(agent) => agent.clone(),
                })))
                .child(
                    List::new()
                        .child(
                            ListBulletItem::new("")
                                .child(Label::new("Create one by visiting"))
                                .child(ButtonLink::new("Zhipu AI Console", "https://open.bigmodel.cn/"))
                        )
                        .child(
                            ListBulletItem::new("Paste your API key below and hit enter to start using the agent")
                        )
                )
                .child(self.api_key_editor.clone())
                .child(
                    Label::new(
                        format!("You can also assign the {API_KEY_ENV_VAR_NAME} environment variable and restart Zed."),
                    )
                    .size(LabelSize::Small)
                    .color(Color::Muted)
                    .mt_0p5(),
                )
                .into_any_element()
        } else {
            ConfiguredApiCard::new(configured_card_label)
                .disabled(env_var_set)
                .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx)))
                .when(env_var_set, |this| {
                    this.tooltip_label(format!(
                        "To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable."
                    ))
                })
                .into_any_element()
        }
    }
}