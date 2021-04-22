#ifndef __cmap_h__
#define __cmap_h__

#ifdef __view__

namespace cv
{

uint8_t bwr_lut[] = { 255,   0,   0,
                      255,   2,   2,
                      255,   4,   4,
                      255,   6,   6,
                      255,   8,   8,
                      255,  10,  10,
                      255,  12,  12,
                      255,  14,  14,
                      255,  16,  16,
                      255,  18,  18,
                      255,  20,  20,
                      255,  22,  22,
                      255,  24,  24,
                      255,  26,  26,
                      255,  28,  28,
                      255,  30,  30,
                      255,  32,  32,
                      255,  34,  34,
                      255,  36,  36,
                      255,  38,  38,
                      255,  40,  40,
                      255,  42,  42,
                      255,  44,  44,
                      255,  46,  46,
                      255,  48,  48,
                      255,  50,  50,
                      255,  52,  52,
                      255,  54,  54,
                      255,  56,  56,
                      255,  58,  58,
                      255,  60,  60,
                      255,  62,  62,
                      255,  64,  64,
                      255,  65,  65,
                      255,  68,  68,
                      255,  70,  70,
                      255,  72,  72,
                      255,  73,  73,
                      255,  76,  76,
                      255,  78,  78,
                      255,  80,  80,
                      255,  81,  81,
                      255,  84,  84,
                      255,  86,  86,
                      255,  88,  88,
                      255,  89,  89,
                      255,  92,  92,
                      255,  94,  94,
                      255,  96,  96,
                      255,  97,  97,
                      255, 100, 100,
                      255, 102, 102,
                      255, 104, 104,
                      255, 105, 105,
                      255, 108, 108,
                      255, 110, 110,
                      255, 112, 112,
                      255, 113, 113,
                      255, 116, 116,
                      255, 118, 118,
                      255, 120, 120,
                      255, 121, 121,
                      255, 124, 124,
                      255, 126, 126,
                      255, 128, 128,
                      255, 130, 130,
                      255, 131, 131,
                      255, 134, 134,
                      255, 136, 136,
                      255, 138, 138,
                      255, 140, 140,
                      255, 142, 142,
                      255, 144, 144,
                      255, 146, 146,
                      255, 147, 147,
                      255, 150, 150,
                      255, 152, 152,
                      255, 154, 154,
                      255, 156, 156,
                      255, 158, 158,
                      255, 160, 160,
                      255, 162, 162,
                      255, 163, 163,
                      255, 166, 166,
                      255, 168, 168,
                      255, 170, 170,
                      255, 172, 172,
                      255, 174, 174,
                      255, 176, 176,
                      255, 178, 178,
                      255, 179, 179,
                      255, 182, 182,
                      255, 184, 184,
                      255, 186, 186,
                      255, 188, 188,
                      255, 190, 190,
                      255, 192, 192,
                      255, 194, 194,
                      255, 195, 195,
                      255, 198, 198,
                      255, 200, 200,
                      255, 202, 202,
                      255, 204, 204,
                      255, 206, 206,
                      255, 208, 208,
                      255, 210, 210,
                      255, 211, 211,
                      255, 214, 214,
                      255, 216, 216,
                      255, 218, 218,
                      255, 220, 220,
                      255, 222, 222,
                      255, 224, 224,
                      255, 226, 226,
                      255, 227, 227,
                      255, 230, 230,
                      255, 232, 232,
                      255, 234, 234,
                      255, 236, 236,
                      255, 238, 238,
                      255, 240, 240,
                      255, 242, 242,
                      255, 243, 243,
                      255, 246, 246,
                      255, 248, 248,
                      255, 250, 250,
                      255, 252, 252,
                      255, 254, 254,
                      254, 254, 255,
                      252, 252, 255,
                      250, 250, 255,
                      248, 248, 255,
                      246, 246, 255,
                      244, 244, 255,
                      242, 242, 255,
                      240, 240, 255,
                      238, 238, 255,
                      236, 236, 255,
                      234, 234, 255,
                      232, 232, 255,
                      230, 230, 255,
                      228, 228, 255,
                      226, 226, 255,
                      224, 224, 255,
                      222, 222, 255,
                      220, 220, 255,
                      218, 218, 255,
                      216, 216, 255,
                      214, 214, 255,
                      211, 211, 255,
                      210, 210, 255,
                      208, 208, 255,
                      206, 206, 255,
                      204, 204, 255,
                      202, 202, 255,
                      200, 200, 255,
                      198, 198, 255,
                      195, 195, 255,
                      194, 194, 255,
                      192, 192, 255,
                      190, 190, 255,
                      188, 188, 255,
                      186, 186, 255,
                      184, 184, 255,
                      182, 182, 255,
                      179, 179, 255,
                      178, 178, 255,
                      176, 176, 255,
                      174, 174, 255,
                      172, 172, 255,
                      170, 170, 255,
                      168, 168, 255,
                      166, 166, 255,
                      163, 163, 255,
                      162, 162, 255,
                      160, 160, 255,
                      158, 158, 255,
                      156, 156, 255,
                      154, 154, 255,
                      152, 152, 255,
                      150, 150, 255,
                      147, 147, 255,
                      146, 146, 255,
                      144, 144, 255,
                      142, 142, 255,
                      140, 140, 255,
                      138, 138, 255,
                      136, 136, 255,
                      134, 134, 255,
                      131, 131, 255,
                      130, 130, 255,
                      128, 128, 255,
                      126, 126, 255,
                      124, 124, 255,
                      121, 121, 255,
                      120, 120, 255,
                      118, 118, 255,
                      116, 116, 255,
                      113, 113, 255,
                      112, 112, 255,
                      110, 110, 255,
                      108, 108, 255,
                      105, 105, 255,
                      104, 104, 255,
                      102, 102, 255,
                      100, 100, 255,
                       97,  97, 255,
                       96,  96, 255,
                       94,  94, 255,
                       92,  92, 255,
                       89,  89, 255,
                       88,  88, 255,
                       86,  86, 255,
                       84,  84, 255,
                       81,  81, 255,
                       80,  80, 255,
                       78,  78, 255,
                       76,  76, 255,
                       73,  73, 255,
                       72,  72, 255,
                       70,  70, 255,
                       68,  68, 255,
                       65,  65, 255,
                       64,  64, 255,
                       62,  62, 255,
                       60,  60, 255,
                       57,  57, 255,
                       56,  56, 255,
                       54,  54, 255,
                       52,  52, 255,
                       49,  49, 255,
                       48,  48, 255,
                       46,  46, 255,
                       44,  44, 255,
                       41,  41, 255,
                       40,  40, 255,
                       38,  38, 255,
                       36,  36, 255,
                       33,  33, 255,
                       32,  32, 255,
                       30,  30, 255,
                       28,  28, 255,
                       25,  25, 255,
                       24,  24, 255,
                       22,  22, 255,
                       20,  20, 255,
                       17,  17, 255,
                       16,  16, 255,
                       14,  14, 255,
                       12,  12, 255,
                        9,   9, 255,
                        8,   8, 255,
                        6,   6, 255,
                        4,   4, 255,
                        1,   1, 255,
                        0,   0, 255};

static cv :: Mat COLORMAP_BWR(256, 1, CV_8UC3, bwr_lut);

} // end namespace

#endif // __view__


#endif // __cmap_h__
