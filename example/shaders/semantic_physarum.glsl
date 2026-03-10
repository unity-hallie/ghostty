// Semantic Physarum — a slime mold that feeds on the meaning of your terminal.
//
// iChannel0: terminal framebuffer
// iChannel1: feedback (previous frame)
// iChannel2: compute state (physarum agent field, if compute kernel present)
// iChannel7: semantic texture (RGB = unit sphere coords, A = confidence)
//
// Without a model, the hash fallback maps identical text to identical sphere
// positions. Similar commands cluster; the physarum finds the ridgelines.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Terminal text
    vec4 term = texture(iChannel0, uv);

    // Previous frame (temporal memory)
    vec4 prev = texture(iChannel1, uv);

    // Semantic field: RGB = sphere position, A = confidence
    vec4 sem = texture(iChannel7, uv);
    float confidence = sem.a;
    vec3 sphere_pos = sem.rgb;

    // Map sphere position to color via equirectangular unwrap
    // theta (polar) and phi (azimuthal) from the unit sphere coords
    float theta = acos(clamp(sphere_pos.z, -1.0, 1.0));
    float phi = atan(sphere_pos.y, sphere_pos.x);
    vec3 semantic_color = vec3(
        0.5 + 0.5 * sin(phi * 2.0 + iTime * 0.3),
        0.5 + 0.5 * sin(theta * 3.0 - iTime * 0.2),
        0.5 + 0.5 * cos(phi + theta + iTime * 0.1)
    );

    // Semantic density: how much "meaning" is nearby
    // Sample neighbors to get gradient
    vec2 texel = 1.0 / iResolution.xy;
    float sem_l = texture(iChannel7, uv - vec2(texel.x, 0)).a;
    float sem_r = texture(iChannel7, uv + vec2(texel.x, 0)).a;
    float sem_u = texture(iChannel7, uv - vec2(0, texel.y)).a;
    float sem_d = texture(iChannel7, uv + vec2(0, texel.y)).a;
    vec2 sem_gradient = vec2(sem_r - sem_l, sem_d - sem_u);
    float sem_density = confidence;

    // Diffusion: the "slime trail" from previous frame
    // Weighted by semantic density — trails persist longer on meaningful text
    float trail_decay = mix(0.92, 0.98, sem_density);
    vec4 diffused = prev * trail_decay;

    // Add some diffusion blur
    vec4 blur = (
        texture(iChannel1, uv + vec2(texel.x, 0)) +
        texture(iChannel1, uv - vec2(texel.x, 0)) +
        texture(iChannel1, uv + vec2(0, texel.y)) +
        texture(iChannel1, uv - vec2(0, texel.y))
    ) * 0.25;
    diffused = mix(diffused, blur, 0.15);

    // Deposit: semantic regions glow with their sphere-color
    vec3 deposit = semantic_color * confidence * 0.08;

    // Pulse: breathe with time, stronger on high-confidence text
    float pulse = 0.5 + 0.5 * sin(iTime * 1.5 + confidence * 6.283);
    deposit *= mix(0.7, 1.3, pulse);

    // Composite: terminal text + semantic glow + trail memory
    vec3 glow = diffused.rgb + deposit;

    // The semantic gradient creates a subtle directional drift
    // visible as the glow "flowing" toward denser meaning
    vec2 drift = sem_gradient * 0.003;
    vec3 drifted_glow = texture(iChannel1, uv + drift).rgb * trail_decay;
    glow = mix(glow, drifted_glow, 0.3);

    // Final composite: text on top, semantic field underneath
    float text_brightness = dot(term.rgb, vec3(0.299, 0.587, 0.114));
    vec3 result = mix(glow * 0.6, term.rgb, smoothstep(0.05, 0.3, text_brightness));

    // Subtle semantic tint on the text itself
    result = mix(result, result * (0.7 + 0.3 * semantic_color), confidence * 0.3);

    fragColor = vec4(result, 1.0);
}
