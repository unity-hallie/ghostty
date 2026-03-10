// Semantic Aurora — the meaning of your terminal, rendered as northern lights.
//
// The sphere coordinates from the semantic texture drive bands of color
// that flow across the screen like aurora borealis. Dense text creates
// bright curtains; empty space is dark sky. The aurora follows the
// topology of meaning — similar text shares the same spectral band.
//
// iChannel0: terminal framebuffer
// iChannel1: feedback (previous frame)
// iChannel7: semantic texture (RGB = sphere coords, A = confidence)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 term = texture(iChannel0, uv);
    vec4 sem = texture(iChannel7, uv);

    float conf = sem.a;
    vec3 sp = sem.rgb;

    // Derive spectral band from sphere azimuthal angle
    float phi = atan(sp.y, sp.x);       // -pi to pi
    float theta = acos(clamp(sp.z, -1.0, 1.0));  // 0 to pi

    // Aurora band: vertical position modulated by semantic angle
    float band = sin(uv.y * 12.0 + phi * 2.0 + iTime * 0.4);
    band += sin(uv.y * 7.0 - theta * 3.0 + iTime * 0.3) * 0.5;
    band = band * 0.5 + 0.5;  // normalize to 0-1
    band = pow(band, 3.0);     // sharpen into curtains

    // Color from sphere position — each semantic cluster gets its own hue
    float hue = phi / 6.283 + 0.5;  // 0-1
    vec3 aurora_color = vec3(
        0.5 + 0.5 * cos(6.283 * (hue + 0.0)),
        0.5 + 0.5 * cos(6.283 * (hue + 0.33)),
        0.5 + 0.5 * cos(6.283 * (hue + 0.67))
    );

    // Intensity driven by semantic confidence and band position
    float intensity = band * conf * 0.7;

    // Shimmer: high-frequency temporal variation
    float shimmer = 0.8 + 0.2 * sin(uv.x * 50.0 + iTime * 5.0 + phi * 10.0);
    intensity *= shimmer;

    // Vertical fade: aurora strongest in upper portion
    float vertical_fade = smoothstep(0.0, 0.7, 1.0 - uv.y);
    intensity *= mix(0.3, 1.0, vertical_fade);

    vec3 aurora = aurora_color * intensity;

    // Dark sky background
    vec3 sky = vec3(0.02, 0.02, 0.05);

    // Feedback: aurora trails persist and drift upward
    vec2 drift = vec2(0.0, -0.002);  // slow upward drift
    vec3 prev = texture(iChannel1, uv + drift).rgb * 0.93;
    aurora = max(aurora, prev * 0.8);

    // Composite
    vec3 background = sky + aurora;

    // Terminal text rendered crisp on top
    float text_lum = dot(term.rgb, vec3(0.299, 0.587, 0.114));
    vec3 text_tinted = term.rgb * (0.8 + 0.2 * aurora_color);  // subtle tint from aurora
    vec3 result = mix(background, text_tinted, smoothstep(0.03, 0.25, text_lum));

    fragColor = vec4(result, 1.0);
}
