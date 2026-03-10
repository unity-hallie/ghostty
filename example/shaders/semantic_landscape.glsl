// Semantic Landscape — the terminal as terrain.
//
// Each line of text has a position on the unit sphere (from the semantic model
// or hash fallback). This shader unwraps the sphere into a heightmap and
// renders the terminal as a landscape where similar text forms ridges and
// valleys. You're looking at the topology of meaning.
//
// iChannel0: terminal framebuffer
// iChannel1: feedback (previous frame)
// iChannel7: semantic texture (RGB = sphere coords, A = confidence)

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 term = texture(iChannel0, uv);
    vec4 sem = texture(iChannel7, uv);

    vec3 sp = sem.rgb;
    float conf = sem.a;

    // Height from sphere position: use z-coordinate as elevation
    // Rotate slowly so the landscape shifts perspective
    float angle = iTime * 0.05;
    float ca = cos(angle), sa = sin(angle);
    vec3 rotated = vec3(
        sp.x * ca - sp.y * sa,
        sp.x * sa + sp.y * ca,
        sp.z
    );
    float height = rotated.z * conf;

    // Compute local gradient for "lighting"
    vec2 texel = 1.0 / iResolution.xy;
    float h_l = texture(iChannel7, uv - vec2(texel.x * 3.0, 0)).z *
                texture(iChannel7, uv - vec2(texel.x * 3.0, 0)).a;
    float h_r = texture(iChannel7, uv + vec2(texel.x * 3.0, 0)).z *
                texture(iChannel7, uv + vec2(texel.x * 3.0, 0)).a;
    float h_u = texture(iChannel7, uv - vec2(0, texel.y * 3.0)).z *
                texture(iChannel7, uv - vec2(0, texel.y * 3.0)).a;
    float h_d = texture(iChannel7, uv + vec2(0, texel.y * 3.0)).z *
                texture(iChannel7, uv + vec2(0, texel.y * 3.0)).a;

    vec3 normal = normalize(vec3(h_l - h_r, h_u - h_d, 0.3));
    vec3 light_dir = normalize(vec3(0.5, -0.3, 1.0));
    float diffuse = max(dot(normal, light_dir), 0.0);

    // Color palette based on height
    vec3 deep = vec3(0.05, 0.05, 0.15);    // valleys: deep blue
    vec3 mid  = vec3(0.1, 0.3, 0.2);       // mid: dark green
    vec3 high = vec3(0.9, 0.85, 0.7);      // peaks: warm light

    vec3 terrain_color = mix(deep, mid, smoothstep(-0.3, 0.0, height));
    terrain_color = mix(terrain_color, high, smoothstep(0.0, 0.5, height));

    // Apply lighting
    terrain_color *= 0.4 + 0.6 * diffuse;

    // Contour lines (topographic map feel)
    float contour = abs(fract(height * 8.0) - 0.5) * 2.0;
    contour = smoothstep(0.0, 0.08, contour);
    terrain_color *= 0.85 + 0.15 * contour;

    // Blend with terminal text — text floats above the terrain
    float text_lum = dot(term.rgb, vec3(0.299, 0.587, 0.114));
    vec3 result = mix(terrain_color, term.rgb * 1.1, smoothstep(0.05, 0.4, text_lum));

    // Subtle previous-frame feedback for temporal smoothing
    vec3 prev = texture(iChannel1, uv).rgb;
    result = mix(prev, result, 0.3);

    fragColor = vec4(result, 1.0);
}
