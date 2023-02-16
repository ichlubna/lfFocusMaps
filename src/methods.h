#ifndef METHODS_H
#define METHODS_H
enum FocusMethod{ONE_DISTANCE, BRUTE_FORCE};
enum ScanMetric{VARIANCE};
enum ClosestFrames{TOP_LEFT=0, TOP_RIGHT=1, BOTTOM_LEFT=2, BOTTOM_RIGHT=3};
enum IntConstantIDs{IMG_RES_X=0, IMG_RES_Y, COLS, ROWS, GRID_SIZE, DISTANCE_ORDER, SCAN_METRIC, FOCUS_METHOD, FOCUS_METHOD_PARAMETER, CLOSEST_VIEWS, BLOCK_SAMPLING, RANGE, YUV_DISTANCE, INT_CONSTANTS_COUNT};
enum FloatConstantIDs{SPACE, FLOAT_CONSTANTS_COUNT};
enum DataPointersIDs{SURFACES, TEXTURES, WEIGHTS, CLOSEST_WEIGHTS, CLOSEST_COORDS, POINTERS_COUNT};
enum FileNames {FOCUS_MAP=0, RENDER_IMAGE=1};
constexpr int CHANNEL_COUNT{4};
#endif
