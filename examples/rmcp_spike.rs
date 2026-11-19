//! M0 spike: a minimal stdio MCP server answering one tool.
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    schemars, tool, tool_router, ServiceExt,
};
use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EchoRequest {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SpikeServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl SpikeServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router(server_handler)]
impl SpikeServer {
    #[tool(description = "Echo the input text back to the caller")]
    fn echo(&self, Parameters(req): Parameters<EchoRequest>) -> String {
        format!("echo:{}", req.text)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = SpikeServer::new().serve(rmcp::transport::stdio()).await?;
    server.waiting().await?;
    Ok(())
}
