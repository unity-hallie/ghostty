//! Wrapper for a Metal compute pipeline state.
//!
//! A compute pipeline encapsulates a single kernel function compiled into
//! a MTLComputePipelineState. It is used with MTLComputeCommandEncoder to
//! dispatch GPU-side simulation passes (e.g. Physarum slime mold, reaction-
//! diffusion systems) that update state textures before the render pass.
const Self = @This();

const std = @import("std");
const macos = @import("macos");
const objc = @import("objc");

const log = std.log.scoped(.metal);

/// MTLComputePipelineState
state: objc.Object,

/// Recommended threadgroup dimensions derived from the pipeline's
/// maxTotalThreadsPerThreadgroup. We use a square threadgroup capped at 16×16.
threadgroup_width: c_ulong,
threadgroup_height: c_ulong,

/// Initialize a compute pipeline from a pre-compiled MTLLibrary.
/// The kernel function must be named "computeMain" in the library.
pub fn init(device: objc.Object, library: objc.Object) !Self {
    // Look up the kernel function by name.
    const kernel_fn: objc.Object = blk: {
        const name = try macos.foundation.String.createWithBytes(
            "computeMain",
            .utf8,
            false,
        );
        defer name.release();

        const ptr = library.msgSend(
            ?*anyopaque,
            objc.sel("newFunctionWithName:"),
            .{name},
        );
        if (ptr == null) {
            log.err("compute kernel 'computeMain' not found in library", .{});
            return error.MetalFailed;
        }
        break :blk objc.Object.fromId(ptr.?);
    };
    defer kernel_fn.msgSend(void, objc.sel("release"), .{});

    // Create the pipeline state from the kernel function.
    var err: ?*anyopaque = null;
    const state = device.msgSend(
        objc.Object,
        objc.sel("newComputePipelineStateWithFunction:error:"),
        .{ kernel_fn, &err },
    );
    try checkError(err);
    errdefer state.release();

    // Query the maximum threads per threadgroup to pick our tile size.
    // We use a 16×16 square, capped by the hardware limit.
    const max_threads = state.getProperty(c_ulong, "maxTotalThreadsPerThreadgroup");
    // Square root of max_threads, rounded down to nearest power of two ≤ 16.
    const side: c_ulong = @min(16, sqrtFloorPow2(max_threads));

    return .{
        .state = state,
        .threadgroup_width = side,
        .threadgroup_height = side,
    };
}

pub fn deinit(self: *const Self) void {
    self.state.release();
}

/// Integer floor-sqrt clamped to a power of two ≤ input.
/// Used to derive a square threadgroup size from maxTotalThreadsPerThreadgroup.
fn sqrtFloorPow2(n: c_ulong) c_ulong {
    if (n == 0) return 1;
    // Simple integer square root
    var s: c_ulong = 1;
    while (s * s * 4 <= n) s *= 2;
    return s;
}

fn checkError(err_: ?*anyopaque) !void {
    const nserr = objc.Object.fromId(err_ orelse return);
    const str = @as(
        *macos.foundation.String,
        @ptrCast(nserr.getProperty(?*anyopaque, "localizedDescription").?),
    );
    log.err("metal compute pipeline error={s}", .{str.cstringPtr(.ascii).?});
    return error.MetalFailed;
}
