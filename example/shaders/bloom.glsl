// Bloom — bright terminal characters emit light into the surrounding void.
//
// Luminance-threshold bloom: pixels above a brightness cutoff are extracted,
// blurred with a multi-tap radial kernel, and composited back over the
// terminal. The result is that bright text — white, cyan, yellow — glows
// softly into the darkness, while dim text stays dark and crisp.
//
// Tune with the constants below.
//
// iChannel0: terminal framebuffer

// ── tuning ────────────────────────────────────────────────────────────────
const float THRESHOLD   = 0.55;  // luminance above which pixels "glow"
const float KNEE        = 0.15;  // soft-knee width around threshold
const float BLOOM_RADIUS = 0.012; // bloom spread, as fraction of screen height
const float BLOOM_STRENGTH = 1.4; // how bright the bloom is
const int   TAPS        = 16;    // samples per bloom pixel (more = softer)
// ──────────────────────────────────────────────────────────────────────────

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Soft-knee threshold: smoothly extract the bright part of a color.
vec3 brightExtract(vec3 c) {
    float l = luma(c);
    float lo = THRESHOLD - KNEE;
    float hi = THRESHOLD + KNEE;
    float w = smoothstep(lo, hi, l);
    return c * w;
}

// Interleaved gradient noise — break up the circular sample pattern.
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;

    // Base terminal pixel.
    vec4 base = texture(iChannel0, uv);

    // ── bloom accumulation ─────────────────────────────────────────────────
    // Spiral Poisson-like sampling in a disk of radius BLOOM_RADIUS.
    // Each tap extracts only bright pixels so dim glyphs don't bloom.
    vec3 bloom = vec3(0.0);
    float total_weight = 0.0;

    // Rotate sample disk by per-pixel noise to avoid banding.
    float angle_offset = ign(fragCoord) * 6.28318;
    float cos_a = cos(angle_offset);
    float sin_a = sin(angle_offset);

    for (int i = 0; i < TAPS; i++) {
        // Sunflower spiral: even, low-discrepancy coverage of the disk.
        float fi = float(i) + 0.5;
        float r  = sqrt(fi / float(TAPS));          // radial position
        float theta = fi * 2.39996323 + angle_offset; // golden angle + jitter
        vec2 offset = vec2(cos(theta), sin(theta)) * r;

        // Scale to screen space, correct for aspect ratio.
        vec2 tap_uv = uv + offset * vec2(BLOOM_RADIUS / aspect, BLOOM_RADIUS);

        vec3 tap = texture(iChannel0, tap_uv).rgb;
        vec3 bright = brightExtract(tap);

        // Weight: falloff from center.
        float w = 1.0 - r;
        bloom += bright * w;
        total_weight += w;
    }

    bloom /= total_weight;

    // ── composite ─────────────────────────────────────────────────────────
    // Additive bloom over the original terminal image.
    // We preserve the base exactly — bloom only adds, never dims.
    vec3 color = base.rgb + bloom * BLOOM_STRENGTH;

    // Subtle tonemap to prevent clipping without washing out.
    color = color / (1.0 + color * 0.25);

    fragColor = vec4(color, base.a);
}
