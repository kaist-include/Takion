#ifndef CUBBYDNN_IMAGE_HPP
#define CUBBYDNN_IMAGE_HPP

#include <string>
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN
{
	class Image
	{
		public:
			Image();
			int Width() const;
			int Height() const;

			static const int COLOR_RGB = 0;
			static const int COLOR_BGR = 1;

			static Image Load(std::string filename);
			static Image Load(std::string filename, int load_color_type);

			void PrintImageTest(std::ostream& output);

			Tensor ToTensor();

		private:
			static Image load_bmp(std::basic_ifstream<unsigned char>& f);
			static Image load_png(std::basic_ifstream<unsigned char>& f);
			static Image load_jpg(std::basic_ifstream<unsigned char>& f);
			int w, h;
			unsigned char*** image;
			bool image_defined;
	};

}  // namespace CubbyDNN

#endif