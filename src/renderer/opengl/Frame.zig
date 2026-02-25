//! Wrapper for handling render passes.
const Self = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const gl = @import("opengl");

const Renderer = @import("../generic.zig").Renderer(OpenGL);
const OpenGL = @import("../OpenGL.zig");
const Target = @import("Target.zig");
const Texture = @import("Texture.zig");
const RenderPass = @import("RenderPass.zig");

const Health = @import("../../renderer.zig").Health;

const log = std.log.scoped(.opengl);

/// Options for beginning a frame.
pub const Options = struct {};

renderer: *Renderer,
target: *Target,

/// Begin encoding a frame.
pub fn begin(
    opts: Options,
    /// Once the frame has been completed, the `frameCompleted` method
    /// on the renderer is called with the health status of the frame.
    renderer: *Renderer,
    /// The target is presented via the provided renderer's API when completed.
    target: *Target,
) !Self {
    _ = opts;

    return .{
        .renderer = renderer,
        .target = target,
    };
}

/// Add a render pass to this frame with the provided attachments.
/// Returns a RenderPass which allows render steps to be added.
pub inline fn renderPass(
    self: *const Self,
    attachments: []const RenderPass.Options.Attachment,
) RenderPass {
    _ = self;
    return RenderPass.begin(.{ .attachments = attachments });
}

/// Copy the contents of the display target to a texture using
/// glBlitFramebuffer. Used to capture the final shader output into
/// the feedback texture so it's available as iChannel1 on the next frame.
pub inline fn blitTexture(self: *const Self, src: Target, dst: Texture) void {
    _ = self;

    // Bind the source target's FBO for reading.
    const src_bind = src.framebuffer.bind(.read) catch return;
    defer src_bind.unbind();

    // Create a temporary FBO and attach the destination texture for drawing.
    const dst_fbo = gl.Framebuffer.create() catch return;
    defer dst_fbo.destroy();

    const dst_bind = dst_fbo.bind(.draw) catch return;
    defer dst_bind.unbind();

    dst_bind.texture2D(.color0, dst.target, dst.texture, 0) catch return;

    // Blit from source to destination.
    gl.glad.context.BlitFramebuffer.?(
        0,
        0,
        @intCast(src.width),
        @intCast(src.height),
        0,
        0,
        @intCast(dst.width),
        @intCast(dst.height),
        gl.c.GL_COLOR_BUFFER_BIT,
        gl.c.GL_NEAREST,
    );
}

/// Complete this frame and present the target.
///
/// If `sync` is true, this will block until the frame is presented.
///
/// NOTE: For OpenGL, `sync` is ignored and we always block.
pub fn complete(self: *const Self, sync: bool) void {
    _ = sync;
    gl.finish();

    // If there are any GL errors, consider the frame unhealthy.
    const health: Health = if (gl.errors.getError()) .healthy else |_| .unhealthy;

    // If the frame is healthy, present it.
    if (health == .healthy) {
        self.renderer.api.present(self.target.*) catch |err| {
            log.err("Failed to present render target: err={}", .{err});
        };
    }

    // Report the health to the renderer.
    self.renderer.frameCompleted(health);
}
