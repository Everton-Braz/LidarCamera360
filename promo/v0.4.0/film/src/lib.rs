//! LidarCamera360 v0.4.0 release film (1920x1080, 30 fps, ~46 s).
//!
//! Eight scenes, cross-dissolved: Hook, Rig, RsOne, Run, Calib, Clean, Export, EndCard.
//! UI captures and fonts are embedded from `media/`. Footage the user supplies later is read
//! at runtime from `footage/` (see README.md); while a file is missing the scene draws a
//! procedural point cloud instead, so the film always renders.
use fframes::{
    AnimateRuntimeInput, AudioMap, Color, Duration, FFramesContext, FFramesSyncedVideoFrame,
    Frame, Overlap, Scene, Scenes, Svgr, SyncVideoFrameInput, Video,
    animation::{AnimationRuntime, Easing},
    include_media_dir,
};
use std::fmt::Write as _;
use std::sync::LazyLock;

include_media_dir!(pub struct FilmMedia, "media");

pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;

const BG: &str = "#0d1015";
const ACCENT: &str = "#22e3f0";
const TEXT: &str = "#f0f4f8";
const MUTED: &str = "#a9b8c8";
const BORDER: &str = "#22414a";
const WARM: &str = "#ff7a59";
const SANS: &str = "Inter";
const MONO: &str = "JetBrains Mono";

/// Cross-dissolve length between scenes; every scene after the first starts this much early.
const XF: f32 = 0.4;

// Footage slots, looked up in `footage/` at runtime.
const FLYTHROUGH: &str = "3dgs_flythrough.mp4";
const CLOUD_BEFORE: &str = "cloud_before.png";
const CLOUD_AFTER: &str = "cloud_after.png";
const STUDIO_RUN: &str = "studio_run.mp4";

static OUT: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(0.5, &Easing::EaseOut));
static XFADE: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(XF, &Easing::EaseInOut));
static CAM: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(1.3, &Easing::EaseInOut));
static MOVE: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(0.75, &Easing::EaseInOut));
static DRAW: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(0.6, &Easing::EaseInOut));
static WIPE: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(2.2, &Easing::EaseInOut));
static PULSE: LazyLock<AnimationRuntime> = LazyLock::new(|| AnimationRuntime::new(0.6, &Easing::EaseOut));
static SPRING: LazyLock<AnimationRuntime> = LazyLock::new(|| {
    AnimationRuntime::new(2.0, &Easing::Spring { mass: 1.0, stiffness: 150.0, damping: 18.0 })
});

/// 0 to 1 with `runtime`, starting at `at` seconds into the scene.
fn prog(frame: &Frame, at: f32, runtime: &AnimationRuntime) -> f32 {
    frame.animate_runtime(AnimateRuntimeInput { on_second: at, from: 0.0, to: 1.0, animation_runtime: runtime })
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Fades `child` in and lifts it 48 px with a soft spring.
fn enter<'a>(frame: &Frame, at: f32, child: Svgr<'a>) -> Svgr<'a> {
    let opacity = prog(frame, at, &OUT);
    let dy = 48.0 * (1.0 - prog(frame, at, &SPRING));
    fframes::svgr!(<g opacity={opacity} transform={format!("translate(0 {dy:.2})")}>{child}</g>)
}

/// Incoming half of the cross-dissolve: every scene paints its own background, so fading
/// the whole scene in over the previous one is a clean dissolve.
fn dissolve<'a>(frame: &Frame, child: Svgr<'a>) -> Svgr<'a> {
    let opacity = prog(frame, 0.0, &XFADE);
    fframes::svgr!(<g opacity={opacity}>{child}</g>)
}

/// Camera on a 1920x1200 picture: puts picture point (px, py) at the frame centre, zoom `z`.
fn cam(px: f32, py: f32, z: f32) -> String {
    format!("translate({:.2} {:.2}) scale({:.4})", 960.0 - px * z, 540.0 - py * z, z)
}

fn card<'a>(x: f32, y: f32, w: f32, h: f32) -> Svgr<'a> {
    fframes::svgr!(<rect x={x} y={y} width={w} height={h} rx="22" fill={BG} fill-opacity="0.92" stroke={BORDER} stroke-width="2" />)
}

fn sans<'a>(x: f32, y: f32, size: u32, weight: u32, fill: &'a str, text: &'a str) -> Svgr<'a> {
    fframes::svgr!(<text x={x} y={y} font-family={SANS} font-size={size} font-weight={weight} fill={fill}>{text}</text>)
}

fn mono<'a>(x: f32, y: f32, size: u32, fill: &'a str, text: &'a str) -> Svgr<'a> {
    fframes::svgr!(<text x={x} y={y} font-family={MONO} font-size={size} font-weight="500" letter-spacing="2" fill={fill}>{text}</text>)
}

/// Dark backdrop shared by every scene.
fn backdrop<'a>() -> Svgr<'a> {
    fframes::svgr!(<rect width="1920" height="1080" fill={BG} />)
}

// ---------------------------------------------------------------------------------------
// Procedural point clouds (stand-ins until the real footage is in `footage/`).
// ---------------------------------------------------------------------------------------

/// Small deterministic random generator, so every render draws the same cloud.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Ground,
    Facade,
    Person,
}

#[derive(Clone, Copy)]
struct Pt {
    x: f32,
    y: f32,
    z: f32,
    kind: Kind,
}

/// A street: ground, two facades with windows, a few trees, and people (dynamic points).
fn street(seed: u64, length: f32, people: bool) -> Vec<Pt> {
    let mut r = Lcg(seed);
    let mut pts = Vec::with_capacity(5000);
    for _ in 0..2200 {
        pts.push(Pt { x: r.range(-8.5, 8.5), y: 0.0, z: r.range(0.0, length), kind: Kind::Ground });
    }
    for side in [-9.0f32, 9.0] {
        for _ in 0..1300 {
            let z = r.range(0.0, length);
            let y = r.range(0.0, 9.0);
            // Leave gaps for windows so the facades read as buildings.
            let window = (z % 4.0) > 1.4 && (z % 4.0) < 2.8 && (y % 3.0) > 1.0 && (y % 3.0) < 2.2;
            if !window {
                pts.push(Pt { x: side + r.range(-0.08, 0.08), y, z, kind: Kind::Facade });
            }
        }
    }
    // Street trees along both curbs.
    let mut z = 4.0;
    while z < length {
        for side in [-6.6f32, 6.6] {
            for _ in 0..40 {
                let a = r.range(0.0, std::f32::consts::TAU);
                let rad = r.range(0.0, 1.3);
                pts.push(Pt { x: side + a.cos() * rad, y: 3.6 + r.range(-0.9, 0.9), z: z + a.sin() * rad, kind: Kind::Facade });
            }
        }
        z += 9.0;
    }
    if people {
        // Standing people and walkers smeared along their path (what --remove-dynamic removes).
        let spots = [(-2.5, 9.0, 0.0), (1.8, 12.0, 2.2), (3.5, 17.0, 0.0), (-4.2, 21.0, -1.8), (0.6, 26.0, 0.0), (-1.4, 31.0, 2.6)];
        for (px, pz, smear) in spots {
            for _ in 0..130 {
                let h = r.range(0.0, 1.75);
                let width = if h > 1.45 { 0.13 } else { 0.24 };
                let a = r.range(0.0, std::f32::consts::TAU);
                let along = r.range(0.0, 1.0) * smear;
                pts.push(Pt { x: px + a.cos() * width + along, y: h, z: pz + a.sin() * width, kind: Kind::Person });
            }
        }
    }
    pts
}

/// Pinhole projection with the camera at (cx, cy, cz) pitched down by `pitch` radians.
fn project(p: Pt, cx: f32, cy: f32, cz: f32, pitch: f32, focal: f32) -> Option<(f32, f32, f32)> {
    let (dx, dy, dz) = (p.x - cx, p.y - cy, p.z - cz);
    let (s, c) = pitch.sin_cos();
    let yc = dy * c + dz * s;
    let zc = -dy * s + dz * c;
    if zc < 0.6 {
        return None;
    }
    let sx = 960.0 + dx / zc * focal;
    let sy = 540.0 - yc / zc * focal;
    if !(-20.0..1940.0).contains(&sx) || !(-20.0..1100.0).contains(&sy) {
        return None;
    }
    Some((sx, sy, zc))
}

/// Points binned by kind and depth into dot paths: one `<path>` per bin keeps the tree small.
fn cloud_paths<'a>(pts: impl Iterator<Item = (Kind, f32, f32, f32)>) -> Svgr<'a> {
    // [kind][depth band]
    let mut bins: [[String; 3]; 3] = Default::default();
    for (kind, x, y, depth) in pts {
        let band = if depth < 12.0 { 0 } else if depth < 26.0 { 1 } else { 2 };
        let k = kind as usize;
        let _ = write!(bins[k][band], "M{x:.1} {y:.1}h0.1");
    }
    let colors = ["#3d8296", ACCENT, WARM];
    let widths = [5.0f32, 3.4, 2.2];
    let alphas = [[0.75f32, 0.55, 0.35], [0.85, 0.6, 0.38], [1.0, 0.9, 0.8]];
    let mut out = Vec::new();
    for k in 0..3 {
        for b in 0..3 {
            let d = std::mem::take(&mut bins[k][b]);
            if d.is_empty() {
                continue;
            }
            let (color, width, alpha) = (colors[k], widths[b], alphas[k][b]);
            out.push(fframes::svgr!(<path d={d} stroke={color} stroke-width={width} stroke-opacity={alpha} stroke-linecap="round" fill="none" />));
        }
    }
    fframes::svgr!(<g>{out}</g>)
}

// ---------------------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------------------

pub struct FilmVideo<'a> {
    pub media: &'a FilmMedia,
    hook: Hook,
    rig: Rig,
    rsone: RsOne,
    run: Run,
    calib: Calib,
    clean: Clean,
    export: Export,
    endcard: EndCard,
}

impl<'a> FilmVideo<'a> {
    pub fn new(media: &'a FilmMedia, _title: &'a str) -> Self {
        Self {
            media,
            hook: Hook { street: street(7, 90.0, false) },
            rig: Rig,
            rsone: RsOne,
            run: Run,
            calib: Calib,
            clean: Clean::new(),
            export: Export,
            endcard: EndCard,
        }
    }
}

impl std::fmt::Debug for FilmVideo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilmVideo").finish()
    }
}

impl Video for FilmVideo<'_> {
    const FPS: usize = 30;
    const WIDTH: usize = WIDTH;
    const HEIGHT: usize = HEIGHT;
    const BACKGROUND_COLOR: Color = Color::BLACK;

    fn duration(&self) -> Duration<'_> {
        Duration::Auto
    }

    fn audio(&self) -> AudioMap<'_> {
        // Music: put `music.mp3` into `footage/` and replace this with
        // AudioMap::from([fframes::AudioTrack::new("music.mp3", fframes::AudioTimestamp::Second(0.)..fframes::AudioTimestamp::Eof).gain_db(-14.).fade_out(1.5)])
        AudioMap::none()
    }

    fn define_scenes(&self) -> Scenes<'_> {
        Scenes::from(vec![
            &self.hook as &dyn Scene,
            &self.rig,
            &self.rsone,
            &self.run,
            &self.calib,
            &self.clean,
            &self.export,
            &self.endcard,
        ])
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        fframes::svgr!(
            <svg xmlns="http://www.w3.org/2000/svg" width={WIDTH} height={HEIGHT} viewBox="0 0 1920 1080">
                {ctx.render_scenes(&frame)}
            </svg>
        )
    }
}

// ---------------------------------------------------------------------------------------
// 1. Hook: 3DGS fly-through (or a procedural street), "Any rig. Calibrated per capture."
// ---------------------------------------------------------------------------------------

struct Hook {
    street: Vec<Pt>,
}

impl std::fmt::Debug for Hook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Hook")
    }
}

impl Scene for Hook {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(4.0)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let t = frame.seconds();
        let input = SyncVideoFrameInput { start_from: 0.0, looping: true, ..Default::default() };
        let footage = match frame.get_synced_video_frame(ctx, FLYTHROUGH, &input) {
            Some(clip) => {
                let image = clip.into_image();
                fframes::svgr!(<image href={image.href()} x="0" y="0" width="1920" height="1080" preserveAspectRatio="xMidYMid slice" />)
            }
            None => {
                // Fly down the street at 5 m/s with a gentle sway.
                let cz = t * 5.0 - 2.0;
                let cx = (t * 0.6).sin() * 0.8;
                let pts = self.street.iter().filter_map(|p| {
                    let mut p = *p;
                    p.z = (p.z - cz).rem_euclid(90.0) + cz;
                    project(p, cx, 1.7, cz, 0.08, 980.0).map(|(x, y, d)| (p.kind, x, y, d))
                });
                cloud_paths(pts)
            }
        };

        fframes::svgr!(<g>
            {backdrop()}
            {footage}
            <defs>
                <linearGradient id="hookShade" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0" stop-color={BG} stop-opacity="0.88" />
                    <stop offset="0.45" stop-color={BG} stop-opacity="0.6" />
                    <stop offset="0.8" stop-color={BG} stop-opacity="0" />
                </linearGradient>
            </defs>
            <rect width="1920" height="1080" fill="url(#hookShade)" />
            {enter(&frame, 0.1, mono(192.0, 420.0, 30, ACCENT, "LIDARCAMERA360 · v0.4.0"))}
            {enter(&frame, 0.25, fframes::svgr!(<text x="186" y="560" font-family={SANS} font-size="132" font-weight="800" letter-spacing="-3" fill={TEXT}>"Any rig."</text>))}
            {enter(&frame, 0.4, fframes::svgr!(<text x="190" y="672" font-family={SANS} font-size="96" font-weight="700" letter-spacing="-2" fill={ACCENT}>"Calibrated per capture."</text>))}
        </g>)
    }
}

// ---------------------------------------------------------------------------------------
// 2. Rig profiles: punch into the rig editor.
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Rig;

impl Scene for Rig {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(6.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let Some(shot) = ctx.get_image("dark_05_rig_profiles.png") else { return Svgr::empty() };
        let k = prog(&frame, XF + 0.3, &CAM);
        let drift = frame.seconds() * 6.0;
        let (px, py, z) = (lerp(960.0, 700.0, k), lerp(600.0, 560.0, k) + drift * k, lerp(1.0, 1.7, k));
        let mark = prog(&frame, XF + 1.6, &DRAW);
        let stroke = 3.0 / z;

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            <g transform={cam(px, py, z)}>
                <image href={shot.href()} x="0" y="0" width="1920" height="1200" />
                <rect x="102" y="444" width="1186" height="270" rx="8" fill={ACCENT} fill-opacity={0.07 * mark} stroke={ACCENT} stroke-width={stroke} stroke-opacity={mark} />
            </g>
            {enter(&frame, XF + 2.0, fframes::svgr!(<g>
                {card(1112.0, 700.0, 616.0, 250.0)}
                {mono(1152.0, 760.0, 28, ACCENT, "RIG PROFILES V2")}
                {sans(1152.0, 830.0, 48, 700, TEXT, "One file per rig.")}
                {sans(1152.0, 896.0, 34, 500, MUTED, "Scanner, cameras, mount.")}
            </g>))}
        </g>))
    }
}

// ---------------------------------------------------------------------------------------
// 3. Insta360 ONE RS: select the pair in the studio.
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct RsOne;

impl Scene for RsOne {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(7.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let Some(shot) = ctx.get_image("dark_01_unified_studio.png") else { return Svgr::empty() };
        let k = prog(&frame, XF + 0.1, &CAM);
        let (px, py, z) = (lerp(960.0, 560.0, k), lerp(600.0, 330.0, k), lerp(1.0, 2.0, k));
        // The ONE RS checkbox in picture space.
        let (bx, by) = (357.0f32, 421.0f32);
        let (sx, sy) = (960.0 + (bx - px) * z, 540.0 + (by - py) * z);

        let click = XF + 2.15;
        let m = prog(&frame, XF + 1.35, &MOVE);
        let (cx, cy) = (lerp(1320.0, sx + 6.0, m), lerp(980.0, sy + 8.0, m));
        let press = if frame.seconds() > click && frame.seconds() < click + 0.12 { 0.86 } else { 1.0 };
        let pulse = prog(&frame, click, &PULSE);
        let checked = frame.seconds() >= click + 0.05;
        let cursor_in = prog(&frame, XF + 1.2, &OUT);

        let check = checked.then(|| fframes::svgr!(<g transform={format!("translate({bx} {by})")}>
            <rect x="-9" y="-9" width="18" height="18" rx="3" fill={ACCENT} />
            <path d="M-5 0 L-1.5 4 L5.5 -4" stroke={BG} stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round" />
        </g>)).unwrap_or_default();

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            <g transform={cam(px, py, z)}>
                <image href={shot.href()} x="0" y="0" width="1920" height="1200" />
                {check}
            </g>
            <circle cx={sx} cy={sy} r={14.0 + 46.0 * pulse} fill="none" stroke={ACCENT} stroke-width="3" stroke-opacity={if pulse > 0.0 { 1.0 - pulse } else { 0.0 }} />
            <g opacity={cursor_in} transform={format!("translate({cx:.1} {cy:.1}) scale({press})")}>
                <path d="M0 0 L0 38 L10 29 L17 45 L24 42 L17 26 L30 26 Z" fill={TEXT} stroke={BG} stroke-width="2.5" stroke-linejoin="round" />
            </g>
            {enter(&frame, XF + 2.5, fframes::svgr!(<g>
                {card(1060.0, 250.0, 668.0, 580.0)}
                {mono(1104.0, 318.0, 28, ACCENT, "NEW CAMERA")}
                {sans(1104.0, 400.0, 60, 800, TEXT, "Insta360 ONE RS")}
                {sans(1104.0, 466.0, 44, 700, TEXT, "1-inch 360")}
                {sans(1104.0, 546.0, 34, 500, MUTED, "Each lens records its own file.")}
                {sans(1104.0, 592.0, 34, 500, MUTED, "Select the pair: one capture.")}
                <rect x="1104" y="644" width="580" height="60" rx="12" fill="#14232a" stroke={BORDER} stroke-width="2" />
                {mono(1128.0, 684.0, 28, TEXT, "VID_…_00_013.insv")}
                <rect x="1104" y="720" width="580" height="60" rx="12" fill="#14232a" stroke={BORDER} stroke-width="2" />
                {mono(1128.0, 760.0, 28, TEXT, "VID_…_10_013.insv")}
            </g>))}
        </g>))
    }
}

// ---------------------------------------------------------------------------------------
// 3b. One run: the real Studio progress panel from a ONE RS capture, sped up.
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Run;

impl Scene for Run {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(7.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let input = SyncVideoFrameInput { start_from: 0.0, looping: false, ..Default::default() };
        let panel = match frame.get_synced_video_frame(ctx, STUDIO_RUN, &input) {
            Some(clip) => {
                let image = clip.into_image();
                fframes::svgr!(<image href={image.href()} x="0" y="0" width="1920" height="506" />)
            }
            None => Svgr::empty(),
        };
        let lift = 1.0 - prog(&frame, XF, &SPRING);

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            <g transform={format!("translate(0 {:.2})", 300.0 + 40.0 * lift)}>
                {panel}
                <rect x="0" y="0" width="1920" height="506" fill="none" stroke={BORDER} stroke-width="2" />
            </g>
            {enter(&frame, XF + 0.1, mono(120.0, 150.0, 28, ACCENT, "ONE CLICK · SEVEN STEPS"))}
            {enter(&frame, XF + 0.25, sans(116.0, 240.0, 72, 800, TEXT, "From two INSV files to a 3DGS dataset."))}
            {enter(&frame, XF + 0.5, mono(1480.0, 150.0, 28, MUTED, "SPED UP ~100×"))}
            {enter(&frame, XF + 0.8, sans(120.0, 900.0, 40, 600, TEXT, "Real ONE RS capture: 160 s of data, done in 10 min 32 s."))}
            {enter(&frame, XF + 1.0, sans(120.0, 962.0, 34, 500, MUTED, "SLAM, sync, rig calibration, masks, colorization, COLMAP export."))}
        </g>))
    }
}

// ---------------------------------------------------------------------------------------
// 4. Calibration per capture: three steps over the dimmed validation panel.
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Calib;

impl Scene for Calib {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(6.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let Some(shot) = ctx.get_image("dark_05_rig_profiles.png") else { return Svgr::empty() };
        let drift = frame.seconds() * 10.0;
        let steps = [
            ("01", "Time offset and mount", "refined for each capture."),
            ("02", "Kept only if unseen", "frames agree better."),
            ("03", "Implausible rigs and", "wrong sizes are refused."),
        ];
        let cards: Vec<Svgr> = steps
            .into_iter()
            .enumerate()
            .map(|(i, (n, a, b))| {
                let x = 192.0 + i as f32 * 520.0;
                enter(&frame, XF + 0.9 + i as f32 * 0.5, fframes::svgr!(<g>
                    {card(x, 520.0, 496.0, 300.0)}
                    {mono(x + 40.0, 600.0, 40, ACCENT, n)}
                    <rect x={x + 40.0} y="628" width="56" height="3" fill={ACCENT} />
                    {sans(x + 40.0, 704.0, 36, 600, TEXT, a)}
                    {sans(x + 40.0, 754.0, 36, 500, MUTED, b)}
                </g>))
            })
            .collect();

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            <g opacity="0.3" transform={cam(940.0 + drift, 900.0, 1.25)}>
                <image href={shot.href()} x="0" y="0" width="1920" height="1200" />
            </g>
            <defs>
                <linearGradient id="calibShade" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stop-color={BG} stop-opacity="0.85" />
                    <stop offset="1" stop-color={BG} stop-opacity="0.35" />
                </linearGradient>
            </defs>
            <rect width="1920" height="1080" fill="url(#calibShade)" />
            {enter(&frame, XF + 0.15, mono(192.0, 270.0, 30, ACCENT, "SELF-CALIBRATION"))}
            {enter(&frame, XF + 0.3, fframes::svgr!(<text x="188" y="400" font-family={SANS} font-size="104" font-weight="800" letter-spacing="-3" fill={TEXT}>"Calibrated per capture."</text>))}
            {cards}
        </g>))
    }
}

// ---------------------------------------------------------------------------------------
// 5. Clean clouds: before/after wipe of --remove-dynamic.
// ---------------------------------------------------------------------------------------

struct Clean {
    before: Vec<Pt>,
}

impl Clean {
    fn new() -> Self {
        Self { before: street(11, 60.0, true) }
    }
}

impl std::fmt::Debug for Clean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Clean")
    }
}

impl Clean {
    fn procedural<'a>(&self, t: f32, people: bool) -> Svgr<'a> {
        // A slow orbit above the street.
        let cx = -1.5 + t * 0.35;
        let pts = self
            .before
            .iter()
            .filter(|p| people || !matches!(p.kind, Kind::Person))
            .filter_map(|p| project(*p, cx, 6.5, -6.0, 0.36, 900.0).map(|(x, y, d)| (p.kind, x, y, d)));
        cloud_paths(pts)
    }
}

impl Scene for Clean {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(7.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let t = frame.seconds();
        let (before, after) = match (ctx.get_image(CLOUD_BEFORE), ctx.get_image(CLOUD_AFTER)) {
            (Some(b), Some(a)) => (
                fframes::svgr!(<image href={b.href()} x="0" y="0" width="1920" height="1080" preserveAspectRatio="xMidYMid slice" />),
                fframes::svgr!(<image href={a.href()} x="0" y="0" width="1920" height="1080" preserveAspectRatio="xMidYMid slice" />),
            ),
            _ => (self.procedural(t, true), self.procedural(t, false)),
        };
        let w = prog(&frame, XF + 0.9, &WIPE);
        let line_x = lerp(1920.0, 0.0, w);
        let line_on = if w > 0.0 && w < 1.0 { 1.0 } else { 0.0 };
        let tag_after = prog(&frame, XF + 1.0, &OUT);
        // The "Before" tag fades out as the wipe line sweeps across it.
        let before_on = ((line_x - 192.0) / 200.0).clamp(0.0, 1.0);

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            {before}
            <defs>
                <clipPath id="afterClip">
                    <rect x={line_x} y="0" width={(1920.0 - line_x).max(0.5)} height="1080" />
                </clipPath>
            </defs>
            <g clip-path="url(#afterClip)">
                {backdrop()}
                {after}
            </g>
            <rect x={line_x - 2.0} y="0" width="4" height="1080" fill={ACCENT} opacity={line_on} />
            <g opacity={before_on}>
                {enter(&frame, XF + 0.2, fframes::svgr!(<g>
                    <rect x="192" y="96" width="168" height="56" rx="28" fill={BG} fill-opacity="0.85" stroke={BORDER} stroke-width="2" />
                    {sans(222.0, 134.0, 30, 600, TEXT, "Before")}
                </g>))}
            </g>
            <g opacity={tag_after}>
                <rect x="1368" y="96" width="360" height="56" rx="28" fill={BG} fill-opacity="0.85" stroke={ACCENT} stroke-width="2" />
                {mono(1398.0, 134.0, 28, ACCENT, "--remove-dynamic")}
            </g>
            {enter(&frame, XF + 3.3, fframes::svgr!(<g>
                {card(192.0, 730.0, 720.0, 230.0)}
                {sans(232.0, 806.0, 52, 800, TEXT, "Clean clouds.")}
                {sans(232.0, 862.0, 34, 500, MUTED, "Moving and standing people removed.")}
                {sans(232.0, 910.0, 34, 500, MUTED, "Blurred frames are skipped.")}
            </g>))}
        </g>))
    }
}

// ---------------------------------------------------------------------------------------
// 6. Export: one project package, every output format.
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Export;

impl Scene for Export {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(5.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, _ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let items = ["project.json", "cloud · images", "camera poses", "trajectory"];
        let rows: Vec<Svgr> = items
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let y = 506.0 + i as f32 * 62.0;
                enter(&frame, XF + 0.6 + i as f32 * 0.08, fframes::svgr!(<g>
                    <circle cx="246" cy={y - 11.0} r="5" fill={ACCENT} />
                    {sans(268.0, y, 34, 500, TEXT, s)}
                </g>))
            })
            .collect();
        let outputs = ["PCD", "LAZ / LAS", "GeoJSON", "GPS track", "Trajectory"];
        let pills: Vec<Svgr> = outputs
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let y = 392.0 + i as f32 * 86.0;
                enter(&frame, XF + 1.45 + i as f32 * 0.1, fframes::svgr!(<g>
                    <rect x="1040" y={y} width="420" height="68" rx="34" fill="#14232a" stroke={BORDER} stroke-width="2" />
                    {sans(1080.0, y + 45.0, 34, 600, TEXT, s)}
                </g>))
            })
            .collect();
        let arrow = prog(&frame, XF + 1.1, &DRAW);
        let len = 190.0f32;

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            {enter(&frame, XF + 0.1, fframes::svgr!(<text x="188" y="250" font-family={SANS} font-size="104" font-weight="800" letter-spacing="-3" fill={TEXT}>"One project. Every format."</text>))}
            {enter(&frame, XF + 0.4, fframes::svgr!(<g>
                {card(192.0, 380.0, 600.0, 420.0)}
                {mono(232.0, 444.0, 34, ACCENT, "project.lc360")}
            </g>))}
            {rows}
            <path d="M820 590 H1010" stroke={ACCENT} stroke-width="4" stroke-dasharray={len} stroke-dashoffset={len * (1.0 - arrow)} fill="none" />
            <path d="M992 574 L1012 590 L992 606" stroke={ACCENT} stroke-width="4" fill="none" stroke-linejoin="round" opacity={if arrow >= 1.0 { 1.0 } else { 0.0 }} />
            {pills}
            {enter(&frame, XF + 2.2, mono(192.0, 900.0, 34, MUTED, "$ lidarcamera360 export <project>"))}
        </g>))
    }
}

// ---------------------------------------------------------------------------------------
// 7. End card.
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct EndCard;

impl Scene for EndCard {
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(4.0)
    }

    fn overlap(&self) -> Overlap {
        Overlap::Previous(XF)
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let logo = ctx.get_image("logo.png").map(|l| {
            fframes::svgr!(<image href={l.href()} x="860" y="180" width="200" height="200" />)
        }).unwrap_or_default();
        let glow = 0.35 + 0.1 * (frame.seconds() * 1.4).sin();

        dissolve(&frame, fframes::svgr!(<g>
            {backdrop()}
            <defs>
                <radialGradient id="endGlow" cx="0.5" cy="0.42" r="0.5">
                    <stop offset="0" stop-color={ACCENT} stop-opacity={glow * 0.35} />
                    <stop offset="1" stop-color={ACCENT} stop-opacity="0" />
                </radialGradient>
            </defs>
            <rect width="1920" height="1080" fill="url(#endGlow)" />
            {enter(&frame, XF + 0.05, fframes::svgr!(<g>{logo}</g>))}
            {enter(&frame, XF + 0.2, fframes::svgr!(<text x="960" y="530" text-anchor="middle" font-family={SANS} font-size="112" font-weight="800" letter-spacing="-3" fill={TEXT}>"LidarCamera360"</text>))}
            {enter(&frame, XF + 0.32, fframes::svgr!(<g>
                <rect x="880" y="566" width="160" height="56" rx="28" fill="none" stroke={ACCENT} stroke-width="2" />
                <text x="960" y="604" text-anchor="middle" font-family={MONO} font-size="30" font-weight="700" fill={ACCENT}>"v0.4.0"</text>
            </g>))}
            {enter(&frame, XF + 0.6, fframes::svgr!(<text x="960" y="740" text-anchor="middle" font-family={MONO} font-size="38" font-weight="500" fill={TEXT}>"github.com/Everton-Braz/LidarCamera360"</text>))}
            {enter(&frame, XF + 0.72, fframes::svgr!(<text x="960" y="808" text-anchor="middle" font-family={SANS} font-size="34" font-weight="500" fill={MUTED}>"Free for Windows · zip or single portable .exe"</text>))}
        </g>))
    }
}
