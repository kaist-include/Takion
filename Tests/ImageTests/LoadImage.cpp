#include <cubbydnn/Preprocess/Image.hpp>
#include <iostream>

int main()
{

    CubbyDNN::Image image = CubbyDNN::Image::Load("C:\\test.bmp");
    image.PrintImageTest(std::cout);
	return 0;
}