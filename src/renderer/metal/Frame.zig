//! Wrapper for handling render passes.
const Self = @This();

const std = @import("std");
const Allocator = std.mem.Allocator;
const objc = @import("objc");

const mtl = @import("api.zig");
const Renderer = @import("../generic.zig").Renderer(Metal);
const Metal = @import("../Metal.zig");
const Target = @import("Target.zig");
const Texture = @import("Texture.zig");
const ComputePipeline = @import("ComputePipeline.zig");
const RenderPass = @import("RenderPass.zig");

const Health = @import("../../renderer.zig").Health;

const log = std.log.scoped(.metal);

/// Options for beginning a frame.
pub const Options = struct {
    /// MTLCommandQueue
    queue: objc.Object,
};

/// MTLCommandBuffer
buffer: objc.Object,

block: CompletionBlock.Context,

/// Begin encoding a frame.
pub fn begin(
    opts: Options,
    /// Once the frame has been completed, the `frameCompleted` method
    /// on the renderer is called with the health status of the frame.
    renderer: *Renderer,
    /// The target is presented via the provided renderer's API when completed.
    target: *Target,
) !Self {
    const buffer = opts.queue.msgSend(
        objc.Object,
        objc.sel("commandBuffer"),
        .{},
    );

    // Create our block to register for completion updates.
    // The block is deallocated by the objC runtime on success.
    const block = CompletionBlock.init(
        .{
            .renderer = renderer,
            .target = target,
            .sync = false,
        },
        &bufferCompleted,
    );

    return .{ .buffer = buffer, .block = block };
}

/// This is the block type used for the addCompletedHandler callback.
const CompletionBlock = objc.Block(struct {
    renderer: *Renderer,
    target: *Target,
    sync: bool,
}, .{
    objc.c.id, // MTLCommandBuffer
}, void);

fn bufferCompleted(
    block: *const CompletionBlock.Context,
    buffer_id: objc.c.id,
) callconv(.c) void {
    const buffer = objc.Object.fromId(buffer_id);

    // Get our command buffer status to pass back to the generic renderer.
    const status = buffer.getProperty(mtl.MTLCommandBufferStatus, "status");
    const health: Health = switch (status) {
        .@"error" => .unhealthy,
        else => .healthy,
    };

    // If the frame is healthy, present it.
    if (health == .healthy) {
        block.renderer.api.present(
            block.target.*,
            block.sync,
        ) catch |err| {
            log.err("Failed to present render target: err={}", .{err});
        };
    }

    block.renderer.frameCompleted(health);
}

/// Add a render pass to this frame with the provided attachments.
/// Returns a RenderPass which allows render steps to be added.
pub inline fn renderPass(
    self: *const Self,
    attachments: []const RenderPass.Options.Attachment,
) RenderPass {
    return RenderPass.begin(.{
        .attachments = attachments,
        .command_buffer = self.buffer,
    });
}

/// Dispatch a compute kernel to update simulation state textures.
/// Must be called BEFORE any render passes that read the state_write texture.
///
/// The compute kernel reads from `textures_read` (bound at indices 0,1,2,...)
/// and writes to `state_write` (bound at index 8, as a write-capable texture).
/// Metal's default hazard tracking ensures the subsequent render pass sees
/// the completed compute writes without manual barriers.
pub const ComputeArgs = struct {
    pipeline: ComputePipeline,
    /// Read-only textures bound at Metal indices 0, 1, 2, ...
    textures_read: []const ?Texture,
    /// The single write-only state texture.
    state_write: Texture,
    /// Metal texture index for the write target (8 for first kernel, 9 for second, etc.).
    state_write_index: usize = 8,
    /// Dispatch dimensions in pixels.
    width: usize,
    height: usize,
};

pub inline fn computePass(self: *const Self, args: ComputeArgs) void {
    const encoder = self.buffer.msgSend(
        objc.Object,
        objc.sel("computeCommandEncoder"),
        .{},
    );
    defer encoder.msgSend(void, objc.sel("endEncoding"), .{});

    encoder.msgSend(void, objc.sel("setComputePipelineState:"), .{args.pipeline.state.value});

    // Bind read textures at their natural indices (0 = iChannel0, 2 = iChannel1, etc.)
    for (args.textures_read, 0..) |maybe_tex, i| {
        if (maybe_tex) |tex| {
            encoder.msgSend(void, objc.sel("setTexture:atIndex:"), .{
                tex.texture.value,
                @as(c_ulong, i),
            });
        }
    }

    // Bind the writable state texture at the specified index (well above any sampled inputs).
    encoder.msgSend(void, objc.sel("setTexture:atIndex:"), .{
        args.state_write.texture.value,
        @as(c_ulong, args.state_write_index),
    });

    // Dispatch: cover every pixel with 16×16 threadgroups.
    const tw = args.pipeline.threadgroup_width;
    const th = args.pipeline.threadgroup_height;
    const threads_per_group = mtl.MTLSize{ .width = tw, .height = th, .depth = 1 };
    const threadgroups = mtl.MTLSize{
        .width  = (args.width  + tw - 1) / tw,
        .height = (args.height + th - 1) / th,
        .depth  = 1,
    };
    encoder.msgSend(void,
        objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"),
        .{ threadgroups, threads_per_group },
    );
}

/// Copy the contents of the display target to a texture using a blit
/// command encoder. Used to capture the final shader output into the
/// feedback texture so it's available as iChannel1 on the next frame.
pub inline fn blitTexture(self: *const Self, src: Target, dst: Texture) void {
    const encoder = self.buffer.msgSend(
        objc.Object,
        objc.sel("blitCommandEncoder"),
        .{},
    );

    // copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
    //   toTexture:destinationSlice:destinationLevel:destinationOrigin:
    encoder.msgSend(void, objc.sel(
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" ++
            "toTexture:destinationSlice:destinationLevel:destinationOrigin:",
    ), .{
        src.texture.value,
        @as(c_ulong, 0),
        @as(c_ulong, 0),
        mtl.MTLOrigin{ .x = 0, .y = 0, .z = 0 },
        mtl.MTLSize{
            .width = @intCast(dst.width),
            .height = @intCast(dst.height),
            .depth = 1,
        },
        dst.texture.value,
        @as(c_ulong, 0),
        @as(c_ulong, 0),
        mtl.MTLOrigin{ .x = 0, .y = 0, .z = 0 },
    });

    encoder.msgSend(void, objc.sel("endEncoding"), .{});
}

/// Complete this frame and present the target.
///
/// If `sync` is true, this will block until the frame is presented.
pub inline fn complete(self: *Self, sync: bool) void {
    // If we don't need to complete synchronously,
    // we add our block as a completion handler.
    //
    // It will be copied when we add the handler, and then the
    // copy will be deallocated by the objc runtime on success.
    if (!sync) {
        self.buffer.msgSend(
            void,
            objc.sel("addCompletedHandler:"),
            .{&self.block},
        );
    }

    self.buffer.msgSend(void, objc.sel("commit"), .{});

    // If we need to complete synchronously, we wait until
    // the buffer is completed and invoke the block directly.
    if (sync) {
        self.buffer.msgSend(void, "waitUntilCompleted", .{});
        self.block.sync = true;
        CompletionBlock.invoke(&self.block, .{self.buffer.value});
    }
}
