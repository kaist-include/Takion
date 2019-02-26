//
// Created by jwkim98 on 19/02/19.
//

#ifndef CUBBYDNN_ENDIAN_HPP
#define CUBBYDNN_ENDIAN_HPP

#if defined(_MSC_VER) && !defined(__clang__)
#define __ORDER_LITTLE_ENDIAN__ 0x4d2
#define __ORDER_BIG_ENDIAN__ 0x10e1
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace CubbnDNN
{
namespace ByteOrder
{
constexpr bool LittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
}
}  // namespace CubbnDNN

#endif  // CUBBYDNN_ENDIAN_HPP
