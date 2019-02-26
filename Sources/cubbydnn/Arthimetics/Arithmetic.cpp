//
// Created by jwkim98 on 19/02/19.
//

#include "../../../Includes/cubbydnn/Arthimetics/Arthimetic.hpp"

namespace CubbyDNN
{
template <typename T>
T CpuArthimatic::Exponential(T input)
{
    return static_cast<T>(exp(input));
}

template <typename T>
T CpuArthimatic::Sin(T input)
{
    return static_cast<T>(sin(input));
}

template <typename T>
T CpuArthimatic::Cos(T input)
{
    return static_cast<T>(cos(input));
}

template <typename T>
T CpuArthimatic::Tan(T input)
{
    return static_cast<T>(tan(input));
}

template <typename T>
T CpuArthimatic::Tanh(T input)
{
    return static_cast<T>(tanh(input));
}

template <typename T>
T CpuArthimatic::Arctan(T input)
{
    return static_cast<T>(atan(input));
}

template <typename T>
T CpuArthimatic::Square(T input)
{
    return static_cast<T>(pow(input, 2));
}

bFloat16 CpuArthimatic::Exponential(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(exp(convertedInput));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

bFloat16 CpuArthimatic::Sin(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(sin(convertedInput));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

bFloat16 CpuArthimatic::Cos(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(cos(convertedInput));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

bFloat16 CpuArthimatic::Tan(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(tan(convertedInput));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

bFloat16 CpuArthimatic::Tanh(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(tanh(convertedInput));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

bFloat16 CpuArthimatic::Arctan(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(atan(convertedInput));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

bFloat16 CpuArthimatic::Square(bFloat16 input)
{
    float convertedInput;
    bFloat16ToFloat(&input, &convertedInput);
    float ans = static_cast<float>(pow(convertedInput, 2));
    bFloat16 result;
    FloatTobFloat16(&ans, &result);
    return result;
}

}  // namespace CubbyDNN
