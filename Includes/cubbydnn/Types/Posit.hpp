/**
 * Include external posit library
 */

#ifndef CUBBYDNN_POSIT_HPP
#define CUBBYDNN_POSIT_HPP

#include "../../../Libraries/universal/posit/posit"
namespace CubbyDNN
{
typedef sw::unum::posit<8, 0> posit8;

typedef sw::unum::posit<16, 1> posit16;

typedef sw::unum::posit<32, 2> posit32;

typedef sw::unum::posit<64, 3> posit64;

}  // namespace CubbyDNN

#endif  // CUBBYDNN_POSIT_HPP
