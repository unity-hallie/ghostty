const std = @import("std");

const Parser = @import("../../osc.zig").Parser;
const Command = @import("../../osc.zig").Command;

/// Parse OSC 7727: set per-surface custom shader.
/// Format: OSC 7727 ; <path> ST
/// An empty path clears the per-surface override.
pub fn parse(parser: *Parser, _: ?u8) ?*Command {
    const writer = parser.writer orelse {
        parser.state = .invalid;
        return null;
    };
    writer.writeByte(0) catch {
        parser.state = .invalid;
        return null;
    };
    const data = writer.buffered();
    parser.command = .{
        .set_shader = .{ .value = data[0 .. data.len - 1 :0] },
    };
    return &parser.command;
}
