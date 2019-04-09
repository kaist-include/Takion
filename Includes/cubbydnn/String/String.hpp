#include <string>

#ifndef CUBBYDNN_STRING_HPP
#define CUBBYDNN_STRING_HPP

namespace CubbyDNN
{
class String
{
 public:
     String();
     String(String str);
     String(char* str);
     ~String();

     std::size_t DataSize();
     bool IsEmpty();

     void Resize();
     void Clear();
     String Substr(unsigned int, unsigned int);

     char& operator[] (unsigned int);

  protected:
     char m_str[32];
};
} //namespace CubbyDNN

#endif //CUBBYDNN_STRING_HPP