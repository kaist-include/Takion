//
// Created by jwkim98 on 19/02/19.
//

#include "cubbydnn/Arthimetics/Arithmetic.hpp"
namespace CubbyDNN
{
template <typename T>
T CpuArithmetic::Exponential(T input)
{
    return static_cast<T>(exp(input));
}

template <typename T>
T CpuArithmetic::Sin(T input)
{
    return static_cast<T>(sin(input));
}

template <typename T>
T CpuArithmetic::Cos(T input)
{
    return static_cast<T>(cos(input));
}

template <typename T>
T CpuArithmetic::Tan(T input)
{
    return static_cast<T>(tan(input));
}

template <typename T>
T CpuArithmetic::Tanh(T input)
{
    return static_cast<T>(tanh(input));
}

template <typename T>
T CpuArithmetic::Arctan(T input)
{
    return static_cast<T>(atan(input));
}

template <typename T>
T CpuArithmetic::Square(T input)
{
    return static_cast<T>(pow(input, 2));
}

template <typename T>
T CpuArithmetic::Log(T input)
{
    return static_cast<T>(log(input));
}

bFloat16 CpuArithmetic::Exponential(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = expf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Sin(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = sinf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Cos(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = cosf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Tan(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = tanf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Tanh(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = tanhf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Arctan(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = atanf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Square(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = powf32(Input32, 2);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

bFloat16 CpuArithmetic::Log(bFloat16 input)
{
    float Input32 = input.getFloat();
    float ans = logf32(Input32);
    uint16_t result;
    TruncateFloatToBfloat16(&ans, &result);
    return bFloat16{ result };
}

}  // namespace CubbyDNN
