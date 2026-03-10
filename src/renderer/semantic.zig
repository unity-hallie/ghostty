//! Semantic texture pipeline for custom shaders.
//!
//! This module extracts visible terminal text, runs it through a sentence
//! embedding model (CoreML on macOS), projects the high-dimensional embeddings
//! onto the unit sphere in 3D, and packs the result into a grid-sized texture
//! that shaders can sample as iChannel7.
//!
//! The pipeline runs on a background thread and writes results to a staging
//! buffer. The renderer picks up new data when available (lock-free flag).
//!
//! Texture format: rgba16float, grid_cols × grid_rows
//!   R,G,B = unit sphere coordinates (semantic position)
//!   A     = confidence / text density (0 = empty cell, 1 = full line)

const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const terminal = @import("../terminal/main.zig");
const page = @import("../terminal/page.zig");
const render = @import("../terminal/render.zig");

const log = std.log.scoped(.semantic);

/// Maximum embedding dimensions we support from the model.
const max_embed_dim = 1024;

/// The 3D projection matrix: embed_dim × 3, stored column-major.
/// Computed via PCA on the first batch of embeddings, then frozen.
const ProjectionMatrix = struct {
    data: [max_embed_dim * 3]f32 = undefined,
    embed_dim: usize = 0,
    ready: bool = false,

    /// Project a high-dimensional embedding to 3D unit sphere.
    /// Returns [3]f32 normalized to unit length.
    fn project(self: *const ProjectionMatrix, embedding: []const f32) [3]f32 {
        var result: [3]f32 = .{ 0, 0, 0 };
        const dim = @min(embedding.len, self.embed_dim);
        for (0..3) |axis| {
            var sum: f32 = 0;
            for (0..dim) |i| {
                sum += embedding[i] * self.data[axis * max_embed_dim + i];
            }
            result[axis] = sum;
        }
        // Normalize to unit sphere
        const len = @sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
        if (len > 1e-8) {
            result[0] /= len;
            result[1] /= len;
            result[2] /= len;
        }
        return result;
    }

    /// Initialize with a random projection (Johnson-Lindenstrauss).
    /// Preserves pairwise distances approximately. Deterministic seed.
    fn initRandom(self: *ProjectionMatrix, embed_dim: usize) void {
        self.embed_dim = embed_dim;
        // Use a fixed seed for deterministic projection
        var rng = std.Random.DefaultPrng.init(0x5E_MA_NT_1C);
        const random = rng.random();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(embed_dim)));
        for (0..3) |axis| {
            for (0..embed_dim) |i| {
                // Random Gaussian approximation via Box-Muller
                const u1 = random.float(f32);
                const u2 = random.float(f32);
                const g = @sqrt(-2.0 * @log(u1 + 1e-10)) * @cos(2.0 * std.math.pi * u2);
                self.data[axis * max_embed_dim + i] = g * scale;
            }
        }
        self.ready = true;
    }
};

/// Staging buffer for semantic texture data. Written by background thread,
/// read by renderer. Double-buffered with atomic flag.
pub const StagingBuffer = struct {
    /// Pixel data: 4 × f16 per cell (RGBA), row-major.
    /// Layout: [rows][cols][4]f16
    data: []u8 = &.{},
    rows: usize = 0,
    cols: usize = 0,

    /// Set atomically by the writer, cleared by the reader.
    new_data_available: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    /// Mutex protecting the data buffer during resize/write.
    mutex: std.Thread.Mutex = .{},

    pub fn init(alloc: Allocator, rows: usize, cols: usize) !StagingBuffer {
        const size = rows * cols * 4 * @sizeOf(f16);
        const data = try alloc.alloc(u8, size);
        @memset(data, 0);
        return .{
            .data = data,
            .rows = rows,
            .cols = cols,
        };
    }

    pub fn deinit(self: *StagingBuffer, alloc: Allocator) void {
        if (self.data.len > 0) alloc.free(self.data);
        self.* = .{};
    }

    pub fn resize(self: *StagingBuffer, alloc: Allocator, rows: usize, cols: usize) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.data.len > 0) alloc.free(self.data);
        const size = rows * cols * 4 * @sizeOf(f16);
        self.data = try alloc.alloc(u8, size);
        @memset(self.data, 0);
        self.rows = rows;
        self.cols = cols;
        self.new_data_available.store(false, .release);
    }

    /// Write a row of semantic data. Called from background thread.
    fn writeRow(self: *StagingBuffer, row: usize, semantic: [3]f32, confidence: f32) void {
        if (row >= self.rows) return;
        const row_offset = row * self.cols * 4 * @sizeOf(f16);
        for (0..self.cols) |col| {
            const offset = row_offset + col * 4 * @sizeOf(f16);
            if (offset + 4 * @sizeOf(f16) > self.data.len) return;
            const pixel: *[4]f16 = @ptrCast(@alignCast(self.data.ptr + offset));
            pixel[0] = @floatCast(semantic[0]);
            pixel[1] = @floatCast(semantic[1]);
            pixel[2] = @floatCast(semantic[2]);
            pixel[3] = @floatCast(confidence);
        }
    }
};

/// Extract visible text lines from terminal render state.
/// Returns a slice of lines (allocated with the provided allocator).
pub fn extractVisibleText(
    alloc: Allocator,
    state: *const terminal.RenderState,
) ![][]const u8 {
    const rows = state.rows;
    const cols = state.cols;

    var lines = try alloc.alloc([]u8, rows);
    var lines_count: usize = 0;
    errdefer {
        for (lines[0..lines_count]) |line| alloc.free(line);
        alloc.free(lines);
    }

    const row_data = state.row_data.slice();
    const row_cells = row_data.items(.cells);

    for (0..rows) |y| {
        var buf = try alloc.alloc(u8, cols * 4); // UTF-8: up to 4 bytes per char
        var pos: usize = 0;

        if (y < row_cells.len) {
            const cells_slice = row_cells[y].slice();
            const raw_cells = cells_slice.items(.raw);

            for (raw_cells) |cell| {
                const cp: u21 = cell.codepoint();
                if (cp == 0) {
                    // Empty cell — write a space
                    if (pos < buf.len) {
                        buf[pos] = ' ';
                        pos += 1;
                    }
                } else {
                    // Encode codepoint as UTF-8
                    const len = std.unicode.utf8Encode(cp, buf[pos..]) catch 0;
                    pos += len;
                }
            }
        }

        // Trim trailing spaces
        while (pos > 0 and buf[pos - 1] == ' ') pos -= 1;

        lines[lines_count] = try alloc.realloc(buf, pos);
        lines_count += 1;
    }

    return @as([][]const u8, @ptrCast(lines));
}

/// Free lines allocated by extractVisibleText.
pub fn freeVisibleText(alloc: Allocator, lines: [][]const u8) void {
    for (lines) |line| alloc.free(@constCast(line));
    alloc.free(@constCast(lines));
}

/// The background semantic analysis thread state.
pub const SemanticThread = struct {
    alloc: Allocator,
    staging: *StagingBuffer,
    projection: ProjectionMatrix = .{},
    model: ?CoreMLModel = null,
    should_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,

    /// Text lines to process (set by renderer, read by this thread).
    pending_lines: ?[][]const u8 = null,
    pending_rows: usize = 0,
    pending_cols: usize = 0,
    pending_mutex: std.Thread.Mutex = .{},
    pending_cond: std.Thread.Condition = .{},

    pub fn init(alloc: Allocator, staging: *StagingBuffer, model_path: ?[]const u8) !SemanticThread {
        var self: SemanticThread = .{
            .alloc = alloc,
            .staging = staging,
        };

        if (model_path) |path| {
            self.model = CoreMLModel.init(alloc, path) catch |err| blk: {
                log.warn("failed to load semantic model \"{s}\": {}", .{ path, err });
                break :blk null;
            };
            if (self.model != null) {
                log.info("loaded semantic model from \"{s}\"", .{path});
            }
        }

        return self;
    }

    pub fn deinit(self: *SemanticThread) void {
        self.stop();
        if (self.model) |*m| m.deinit();
        if (self.pending_lines) |lines| freeVisibleText(self.alloc, lines);
    }

    pub fn start(self: *SemanticThread) !void {
        if (self.thread != null) return;
        self.thread = try std.Thread.spawn(.{}, threadMain, .{self});
    }

    pub fn stop(self: *SemanticThread) void {
        if (self.thread) |thread| {
            self.should_stop.store(true, .release);
            self.pending_cond.signal();
            thread.join();
            self.thread = null;
        }
    }

    /// Called by the renderer when terminal content changes.
    /// Passes extracted text lines for the background thread to process.
    pub fn submitLines(self: *SemanticThread, lines: [][]const u8, rows: usize, cols: usize) void {
        self.pending_mutex.lock();
        defer self.pending_mutex.unlock();

        // Free previous pending lines if not yet consumed
        if (self.pending_lines) |old| freeVisibleText(self.alloc, old);
        self.pending_lines = lines;
        self.pending_rows = rows;
        self.pending_cols = cols;
        self.pending_cond.signal();
    }

    fn threadMain(self: *SemanticThread) void {
        while (!self.should_stop.load(.acquire)) {
            // Wait for new lines to process
            self.pending_mutex.lock();
            while (self.pending_lines == null and !self.should_stop.load(.acquire)) {
                self.pending_cond.wait(&self.pending_mutex);
            }

            if (self.should_stop.load(.acquire)) {
                self.pending_mutex.unlock();
                break;
            }

            const lines = self.pending_lines.?;
            const rows = self.pending_rows;
            const cols = self.pending_cols;
            self.pending_lines = null;
            self.pending_mutex.unlock();

            // Process the lines
            self.processLines(lines, rows, cols);
            freeVisibleText(self.alloc, lines);
        }
    }

    fn processLines(self: *SemanticThread, lines: [][]const u8, rows: usize, cols: usize) void {
        _ = cols;

        self.staging.mutex.lock();
        defer self.staging.mutex.unlock();

        for (lines, 0..) |line, row| {
            if (row >= rows) break;

            if (line.len == 0) {
                self.staging.writeRow(row, .{ 0, 0, 0 }, 0);
                continue;
            }

            // Run through model if available, otherwise use hash-based fallback
            const semantic: [3]f32 = if (self.model) |*model| blk: {
                const embedding = model.embed(self.alloc, line) catch {
                    break :blk hashFallback(line);
                };
                defer self.alloc.free(embedding);

                // Initialize projection on first embedding if needed
                if (!self.projection.ready) {
                    self.projection.initRandom(embedding.len);
                }
                break :blk self.projection.project(embedding);
            } else hashFallback(line);

            const confidence: f32 = @min(1.0, @as(f32, @floatFromInt(line.len)) / 40.0);
            self.staging.writeRow(row, semantic, confidence);
        }

        self.staging.new_data_available.store(true, .release);
    }

    /// Deterministic hash-based fallback when no model is loaded.
    /// Maps text to a point on the unit sphere using the hash as angles.
    fn hashFallback(text: []const u8) [3]f32 {
        var h = std.hash.Fnv1a_64.init();
        h.update(text);
        const hash = h.final();

        // Use hash bits as two angles for spherical coordinates
        const theta: f32 = @as(f32, @floatFromInt(hash & 0xFFFFFFFF)) / @as(f32, @floatFromInt(@as(u32, 0xFFFFFFFF))) * std.math.pi;
        const phi: f32 = @as(f32, @floatFromInt(hash >> 32)) / @as(f32, @floatFromInt(@as(u32, 0xFFFFFFFF))) * 2.0 * std.math.pi;

        return .{
            @sin(theta) * @cos(phi),
            @sin(theta) * @sin(phi),
            @cos(theta),
        };
    }
};

/// CoreML model wrapper for sentence embeddings.
/// Uses Obj-C runtime to call CoreML APIs directly.
pub const CoreMLModel = struct {
    /// The compiled MLModel object (retained).
    model: if (builtin.os.tag == .macos) @import("objc").Object else void,

    pub fn init(alloc: Allocator, path: []const u8) !CoreMLModel {
        _ = alloc;
        if (builtin.os.tag != .macos) return error.Unsupported;

        const objc = @import("objc");

        // Create NSURL from path
        const NSString = objc.getClass("NSString").?;
        const path_str = NSString.msgSend(
            objc.Object,
            objc.sel("stringWithUTF8String:"),
            .{path.ptr},
        );

        const NSURL = objc.getClass("NSURL").?;
        const url = NSURL.msgSend(
            objc.Object,
            objc.sel("fileURLWithPath:"),
            .{path_str.value},
        );

        // Load the compiled CoreML model
        const MLModel = objc.getClass("MLModel") orelse {
            log.warn("MLModel class not found — CoreML not available", .{});
            return error.Unsupported;
        };

        var err: ?*anyopaque = null;
        const compiled_url = MLModel.msgSend(
            ?*anyopaque,
            objc.sel("compileModelAtURL:error:"),
            .{ url.value, &err },
        );

        if (compiled_url == null) {
            log.warn("failed to compile CoreML model", .{});
            return error.ModelLoadFailed;
        }

        var load_err: ?*anyopaque = null;
        const model = MLModel.msgSend(
            ?*anyopaque,
            objc.sel("modelWithContentsOfURL:error:"),
            .{ compiled_url.?, &load_err },
        );

        if (model == null) {
            log.warn("failed to load CoreML model", .{});
            return error.ModelLoadFailed;
        }

        const model_obj = objc.Object.fromId(model.?);
        // Retain the model so it survives beyond this scope
        _ = model_obj.msgSend(objc.Object, objc.sel("retain"), .{});

        return .{ .model = model_obj };
    }

    pub fn deinit(self: *CoreMLModel) void {
        if (builtin.os.tag == .macos) {
            self.model.msgSend(void, @import("objc").sel("release"), .{});
        }
    }

    /// Run inference to get an embedding vector for the given text.
    /// Returns a float slice allocated with `alloc`.
    pub fn embed(self: *CoreMLModel, alloc: Allocator, text: []const u8) ![]f32 {
        if (builtin.os.tag != .macos) return error.Unsupported;

        const objc = @import("objc");

        // Create input dictionary with the text
        // This assumes the model has a "text" input of type String.
        // The actual input name depends on the model — we could make this configurable.
        const NSString = objc.getClass("NSString").?;
        const text_str = NSString.msgSend(
            objc.Object,
            objc.sel("stringWithUTF8String:"),
            .{text.ptr},
        );

        const key_str = NSString.msgSend(
            objc.Object,
            objc.sel("stringWithUTF8String:"),
            .{"text".ptr},
        );

        const NSDictionary = objc.getClass("NSDictionary").?;
        const input_dict = NSDictionary.msgSend(
            objc.Object,
            objc.sel("dictionaryWithObject:forKey:"),
            .{ text_str.value, key_str.value },
        );

        // Create MLDictionaryFeatureProvider
        const MLDictProvider = objc.getClass("MLDictionaryFeatureProvider").?;
        var prov_err: ?*anyopaque = null;
        const provider_id = MLDictProvider.msgSend(
            objc.Object,
            objc.sel("alloc"),
            .{},
        ).msgSend(
            ?*anyopaque,
            objc.sel("initWithDictionary:error:"),
            .{ input_dict.value, &prov_err },
        );

        if (provider_id == null) return error.InferenceFailed;
        const provider = objc.Object.fromId(provider_id.?);
        defer provider.msgSend(void, objc.sel("release"), .{});

        // Run prediction
        var pred_err: ?*anyopaque = null;
        const output_id = self.model.msgSend(
            ?*anyopaque,
            objc.sel("predictionFromFeatures:error:"),
            .{ provider.value, &pred_err },
        );

        if (output_id == null) return error.InferenceFailed;
        const output = objc.Object.fromId(output_id.?);

        // Get the embedding output — assume it's a MLMultiArray named "embedding"
        const embed_key = NSString.msgSend(
            objc.Object,
            objc.sel("stringWithUTF8String:"),
            .{"embedding".ptr},
        );

        const multi_array_id = output.msgSend(
            ?*anyopaque,
            objc.sel("featureValueForName:"),
            .{embed_key.value},
        );
        if (multi_array_id == null) return error.InferenceFailed;
        const feature_value = objc.Object.fromId(multi_array_id.?);

        const array_id = feature_value.msgSend(
            ?*anyopaque,
            objc.sel("multiArrayValue"),
            .{},
        );
        if (array_id == null) return error.InferenceFailed;
        const multi_array = objc.Object.fromId(array_id.?);

        // Get the data pointer and count
        const count: usize = @intCast(multi_array.getProperty(c_long, "count"));
        const data_ptr = multi_array.msgSend([*]f32, objc.sel("dataPointer"), .{});

        // Copy to our owned buffer
        const result = try alloc.alloc(f32, count);
        @memcpy(result, data_ptr[0..count]);

        return result;
    }
};
