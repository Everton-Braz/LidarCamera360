use fframes::{CombinedMediaProvider, EncoderOptions, MediaDirectory, MediaProvider, RenderOptions, StaticMediaProvider, cli};
use fframes_skia_renderer::{
    SkiaFFramesRenderer, SkiaPipelineConcurrencyPolicy, SkiaPipelineConfig,
    vulkan::SkiaVulkanCtx,
};
use film::{ FilmMedia, FilmVideo, HEIGHT, WIDTH };
use fframes::cli::clap; // the derive below expands to `clap::...`
use std::process::ExitCode;

/// Flags of this video next to the standard ones of `fframes::cli` (render, frame, strip,
/// inspect, audio, ...). Run `cargo run --release -- --help`.
#[derive(Debug, clap::Args)]
struct VideoArgs {
    #[arg(long, default_value = "LidarCamera360 v0.4.0", global = true)]
    title: String,
}

fn main() -> ExitCode {
    let args = cli::parse::<VideoArgs>();
    let media = FilmMedia::prepare().expect("media");
    // Footage (3DGS fly-through, cloud stills, music) is read at runtime from `footage/`
    // so large files are not compiled into the binary. Missing files fall back in lib.rs.
    let footage_dir = MediaDirectory::read_folder("footage").ok();
    let footage = footage_dir.as_ref().map(|dir| dir.process_media_source().expect("footage"));
    let combined;
    let provider: &dyn MediaProvider = match &footage {
        Some(footage) => {
            combined = CombinedMediaProvider::from([&media as &dyn MediaProvider, footage as &dyn MediaProvider]);
            &combined
        }
        None => &media,
    };
    let title = args.app.title.clone();
    let video = FilmVideo::new(&media, &title);
    let gpu = SkiaVulkanCtx::new(WIDTH, HEIGHT).expect("GPU context");

    cli::new(
        &video,
        RenderOptions {
            media: Some(provider),
            video_encoder_options: EncoderOptions {
                preferred_encoder: Some("libx264"),
                codec_params: Some(&[("crf", "20"), ("preset", "medium"), ("tune", "animation")]),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .args(args)
    // Frame previews (frame, strip, onion, snapshot) render with the same Skia backend.
    .backend(
        SkiaFFramesRenderer::new_vulkan(
            &gpu,
            SkiaPipelineConfig {
                concurrency_policy: SkiaPipelineConcurrencyPolicy::MaxPerformance,
                ..Default::default()
            },
        )
        .expect("skia renderer"),
    )
    // `preview` opens the real-time GPU player.
    .preview(fframes_native_player::cli_preview)
    .run()
}
