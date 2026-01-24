//! Neolith - A high-performance voxel engine
//!
//! Built for scalability from the metal: world generation, chunk management,
//! and rendering performance at scale.

use neolith::App;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // Initialize logging with performance tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("neolith=info".parse().unwrap())
            .add_directive("wgpu=warn".parse().unwrap()))
        .init();

    tracing::info!("Starting Neolith engine...");

    pollster::block_on(App::run());
}
