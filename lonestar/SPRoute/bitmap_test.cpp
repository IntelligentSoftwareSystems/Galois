/*
 *****************************************************************************
 *                                                                           *
 *                          Platform Independent                             *
 *                     Bitmap Image Reader Writer Library                    *
 *                                                                           *
 * Author: Arash Partow - 2002                                               *
 * URL: http://partow.net/programming/bitmap/index.html                      *
 *                                                                           *
 * Note: This library only supports 24-bits per pixel bitmap format files.   *
 *                                                                           *
 * Copyright notice:                                                         *
 * Free use of the Platform Independent Bitmap Image Reader Writer Library   *
 * is permitted under the guidelines and in accordance with the most current *
 * version of the MIT License.                                               *
 * http://www.opensource.org/licenses/MIT                                    *
 *                                                                           *
 *****************************************************************************
*/


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "bitmap_image.hpp"


void test01()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test01() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   image.save_image("test01_saved.bmp");
}

void test02()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test02() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   image.save_image("test02_saved.bmp");

   image.vertical_flip();
   image.save_image("test02_saved_vert_flip.bmp");
   image.vertical_flip();

   image.horizontal_flip();
   image.save_image("test02_saved_horiz_flip.bmp");
}

void test03()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test03() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   bitmap_image subsampled_image1;
   bitmap_image subsampled_image2;
   bitmap_image subsampled_image3;

   image.subsample(subsampled_image1);
   subsampled_image1.save_image("test03_1xsubsampled_image.bmp");

   subsampled_image1.subsample(subsampled_image2);
   subsampled_image2.save_image("test03_2xsubsampled_image.bmp");

   subsampled_image2.subsample(subsampled_image3);
   subsampled_image3.save_image("test03_3xsubsampled_image.bmp");
}

void test04()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test04() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   bitmap_image upsampled_image1;
   bitmap_image upsampled_image2;
   bitmap_image upsampled_image3;

   image.upsample(upsampled_image1);
   upsampled_image1.save_image("test04_1xupsampled_image.bmp");

   upsampled_image1.upsample(upsampled_image2);
   upsampled_image2.save_image("test04_2xupsampled_image.bmp");

   upsampled_image2.upsample(upsampled_image3);
   upsampled_image3.save_image("test04_3xupsampled_image.bmp");
}

void test05()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test05() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   image.set_all_ith_bits_low(0);
   image.save_image("test05_lsb0_removed_saved.bmp");
   image.set_all_ith_bits_low(1);
   image.save_image("test05_lsb01_removed_saved.bmp");
   image.set_all_ith_bits_low(2);
   image.save_image("test05_lsb012_removed_saved.bmp");
   image.set_all_ith_bits_low(3);
   image.save_image("test05_lsb0123_removed_saved.bmp");
   image.set_all_ith_bits_low(4);
   image.save_image("test05_lsb01234_removed_saved.bmp");
   image.set_all_ith_bits_low(5);
   image.save_image("test05_lsb012345_removed_saved.bmp");
   image.set_all_ith_bits_low(6);
   image.save_image("test05_lsb0123456_removed_saved.bmp");
}

void test06()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test06() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   bitmap_image red_channel_image;
   image.export_color_plane(bitmap_image::red_plane,red_channel_image);
   red_channel_image.save_image("test06_red_channel_image.bmp");

   bitmap_image green_channel_image;
   image.export_color_plane(bitmap_image::green_plane,green_channel_image);
   green_channel_image.save_image("test06_green_channel_image.bmp");

   bitmap_image blue_channel_image;
   image.export_color_plane(bitmap_image::blue_plane,blue_channel_image);
   blue_channel_image.save_image("test06_blue_channel_image.bmp");
}

void test07()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test07() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   image.convert_to_grayscale();
   image.save_image("test07_grayscale_image.bmp");
}

void test08()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test08() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   bitmap_image image1;
   bitmap_image image2;
   bitmap_image image3;
   bitmap_image image4;

   unsigned int w = image.width();
   unsigned int h = image.height();

   if (!image.region(0,0, w / 2, h / 2,image1))
   {
      std::cout << "ERROR: upper_left_image" << std::endl;
   }

   if (!image.region((w - 1) / 2, 0, w / 2, h / 2,image2))
   {
      std::cout << "ERROR: upper_right_image" << std::endl;
   }

   if (!image.region(0,(h - 1) / 2, w / 2, h / 2,image3))
   {
      std::cout << "ERROR: lower_left_image" << std::endl;
   }

   if (!image.region((w - 1) / 2, (h - 1) / 2, w / 2, h / 2,image4))
   {
      std::cout << "ERROR: lower_right_image" << std::endl;
   }

   image1.save_image("test08_upper_left_image.bmp");
   image2.save_image("test08_upper_right_image.bmp");
   image3.save_image("test08_lower_left_image.bmp");
   image4.save_image("test08_lower_right_image.bmp");
}

void test09()
{
   const unsigned int dim = 1000;

   bitmap_image image(dim,dim);

   for (unsigned int x = 0; x < dim; ++x)
   {
      for (unsigned int y = 0; y < dim; ++y)
      {
         rgb_t col = jet_colormap[(x + y) % dim];
         image.set_pixel(x,y,col.red,col.green,col.blue);
      }
   }

   image.save_image("test09_color_map_image.bmp");
}

void test10()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test10() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   image.invert_color_planes();
   image.save_image("test10_inverted_color_image.bmp");
}

void test11()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test11() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   for (unsigned int i = 0; i < 10; ++i)
   {
      image.add_to_color_plane(bitmap_image::red_plane,10);
      image.save_image(std::string("test11_") + static_cast<char>(48 + i) + std::string("_red_inc_image.bmp"));
   }
}

void test12()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test12() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   double* y  = new double [image.pixel_count()];
   double* cb = new double [image.pixel_count()];
   double* cr = new double [image.pixel_count()];

   image.export_ycbcr(y,cb,cr);

   for (unsigned int i = 0; i < image.pixel_count(); ++i)
   {
      cb[i] = cr[i] = 0.0;
   }

   image.import_ycbcr(y,cb,cr);
   image.save_image("test12_only_y_image.bmp");

   delete[] y;
   delete[] cb;
   delete[] cr;
}

void test13()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test13() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   double* y  = new double [image.pixel_count()];
   double* cb = new double [image.pixel_count()];
   double* cr = new double [image.pixel_count()];

   image.export_ycbcr(y,cb,cr);

   for (unsigned int j = 0; j < 10; ++j)
   {
      for (unsigned int i = 0; i < image.pixel_count(); ++i)
      {
         y[i] += 15.0;
      }

      image.import_ycbcr(y,cb,cr);
      image.save_image(std::string("test13_") + static_cast<char>(48 + j) + std::string("_y_image.bmp"));
   }

   delete[] y;
   delete[] cb;
   delete[] cr;
}

void test14()
{
   bitmap_image image(512,512);

   image.clear();
   checkered_pattern(64,64,220,bitmap_image::red_plane,image);
   image.save_image("test14_checkered_01.bmp");

   image.clear();
   checkered_pattern(32,64,100,200,50,image);
   image.save_image("test14_checkered_02.bmp");
}

void test15()
{
   bitmap_image image(1024,1024);

   image.clear();

   double c1 = 0.9;
   double c2 = 0.5;
   double c3 = 0.3;
   double c4 = 0.7;

   ::srand(0xA5AA5AA5);
   plasma(image,0,0,image.width(),image.height(),c1,c2,c3,c4,3.0,jet_colormap);
   image.save_image("test15_plasma.bmp");
}

void test16()
{
   std::string file_name("image.bmp");

   bitmap_image image(file_name);

   if (!image)
   {
      printf("test16() - Error - Failed to open '%s'\n",file_name.c_str());
      return;
   }

   double c1 = 0.9;
   double c2 = 0.5;
   double c3 = 0.3;
   double c4 = 0.7;

   bitmap_image plasma_image(image.width(),image.height());
   plasma(plasma_image,0,0,plasma_image.width(),plasma_image.height(),c1,c2,c3,c4,3.0,jet_colormap);

   bitmap_image temp_image(image);

   temp_image.alpha_blend(0.1, plasma_image);
   temp_image.save_image("test16_alpha_0.1.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.2, plasma_image);
   temp_image.save_image("test16_alpha_0.2.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.3, plasma_image);
   temp_image.save_image("test16_alpha_0.3.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.4, plasma_image);
   temp_image.save_image("test16_alpha_0.4.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.5, plasma_image);
   temp_image.save_image("test16_alpha_0.5.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.6, plasma_image);
   temp_image.save_image("test16_alpha_0.6.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.7, plasma_image);
   temp_image.save_image("test16_alpha_0.7.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.8, plasma_image);
   temp_image.save_image("test16_alpha_0.8.bmp");
   temp_image = image;

   temp_image.alpha_blend(0.9, plasma_image);
   temp_image.save_image("test16_alpha_0.9.bmp");
}

void test17()
{
   bitmap_image image(1024,1024);

   double c1 = 0.9;
   double c2 = 0.5;
   double c3 = 0.3;
   double c4 = 0.7;

   plasma(image,0,0,image.width(),image.height(),c1,c2,c3,c4,3.0,jet_colormap);

   image_drawer draw(image);

   draw.pen_width(3);
   draw.pen_color(255,0,0);
   draw.circle(image.width() / 2 + 100, image.height() / 2, 100);

   draw.pen_width(2);
   draw.pen_color(0,255,255);
   draw.ellipse(image.width() / 2, image.height() / 2, 200,350);

   draw.pen_width(1);
   draw.pen_color(255,255,0);
   draw.rectangle(50,50,250,400);

   draw.pen_color(0,255,0);
   draw.rectangle(450,250,850,880);

   image.save_image("test17_image_drawer.bmp");
}

void test18()
{
   {
      bitmap_image image(1000,180);
      image_drawer draw(image);
      const rgb_t* colormap[9] = {
                                   autumn_colormap,
                                   copper_colormap,
                                     gray_colormap,
                                      hot_colormap,
                                      hsv_colormap,
                                      jet_colormap,
                                    prism_colormap,
                                      vga_colormap,
                                     yarg_colormap
                                 };

      for (unsigned int i = 0; i < image.width(); ++i)
      {
         for (unsigned int j = 0; j < 9; ++j)
         {
            draw.pen_color(colormap[j][i].red,colormap[j][i].green,colormap[j][i].blue);
            draw.vertical_line_segment(j * 20, (j + 1) * 20, i);
         }
      }

      image.save_image("test18_color_maps.bmp");
   }

   {
      bitmap_image image(1000,500);
      image_drawer draw(image);

      std::size_t palette_colormap_size = sizeof(palette_colormap) / sizeof(rgb_t);
      std::size_t bar_width = image.width() / palette_colormap_size;

      for (std::size_t i = 0; i < palette_colormap_size; ++i)
      {
         for (std::size_t j = 0; j < bar_width; ++j)
         {
            draw.pen_color(palette_colormap[i].red,palette_colormap[i].green,palette_colormap[i].blue);
            draw.vertical_line_segment(0, image.height(), static_cast<int>(i * bar_width + j));
         }
      }

      image.save_image("test18_palette_colormap.bmp");
   }
}

void test19()
{
   {
      cartesian_canvas canvas(1000, 1000);

      if (!canvas)
      {
         printf("test19() - Error - Failed to instantiate cartesian canvas(1000x1000) [1]\n");
         return;
      }

      canvas.rectangle(canvas.min_x(), canvas.min_y(), canvas.max_x(), canvas.max_y());

      canvas.horiztonal_line_segment(canvas.min_x(), canvas.max_x(), -400.0);

      canvas.line_segment(-500.0, 600.0, 600.0, -500.0);

      canvas.pen_width(3);

      for (std::size_t i = 0; i < 160; i++)
      {
         std::size_t c_idx = i % (sizeof(palette_colormap) / sizeof(rgb_t));

         canvas.pen_color(palette_colormap[c_idx].red, palette_colormap[c_idx].green, palette_colormap[c_idx].blue);

         canvas.circle(0.0, 0.0, 3.0 * i);
      }

      canvas.image().save_image("test19_cartesian_canvas01.bmp");
   }

   {
      static const double pi = 3.14159265358979323846264338327950288419716939937510;

      cartesian_canvas canvas(1000, 1000);

      if (!canvas)
      {
         printf("test19() - Error - Failed to instantiate cartesian canvas(1000x1000) [2]\n");
         return;
      }

      canvas.image().set_all_channels(0xFF);

      canvas.pen_width(2);

      unsigned int i = 0;

      for (double x = -500; x < 500; x += 3, ++i)
      {
         std::size_t c_idx = i % (sizeof(palette_colormap) / sizeof(rgb_t));

         canvas.pen_color(palette_colormap[c_idx].red, palette_colormap[c_idx].green, palette_colormap[c_idx].blue);

         double radius = std::max(10.0,std::abs(80.0 * std::sin((1.0 / 80.0) * pi * x)));

         double y = 400.0 * std::sin((1.0 / 200.0) * pi * x);

         canvas.circle(x, y, radius);
      }

      canvas.image().save_image("test19_cartesian_canvas02.bmp");
   }
}

void test20()
{
   const rgb_t* colormap[4] = {
                                   hsv_colormap,
                                   jet_colormap,
                                 prism_colormap,
                                   vga_colormap
                              };

   const unsigned int fractal_width  = 1200;
   const unsigned int fractal_height =  800;

   {
      bitmap_image fractal_hsv  (fractal_width,fractal_height);
      bitmap_image fractal_jet  (fractal_width,fractal_height);
      bitmap_image fractal_prism(fractal_width,fractal_height);
      bitmap_image fractal_vga  (fractal_width,fractal_height);

      fractal_hsv  .clear();
      fractal_jet  .clear();
      fractal_prism.clear();
      fractal_vga  .clear();

      double    cr,    ci;
      double nextr, nexti;
      double prevr, previ;

      const unsigned int max_iterations = 1000;

      for (unsigned int y = 0; y < fractal_height; ++y)
      {
         for (unsigned int x = 0; x < fractal_width; ++x)
         {
            cr = 1.5 * (2.0 * x / fractal_width  - 1.0) - 0.5;
            ci =       (2.0 * y / fractal_height - 1.0);

            nextr = nexti = 0;
            prevr = previ = 0;

            for (unsigned int i = 0; i < max_iterations; i++)
            {
               prevr = nextr;
               previ = nexti;

               nextr =     prevr * prevr - previ * previ + cr;
               nexti = 2 * prevr * previ + ci;

               if (((nextr * nextr) + (nexti * nexti)) > 4)
               {
                  if (max_iterations != i)
                  {
                     double z = sqrt(nextr * nextr + nexti * nexti);

                     #define log2(x) (std::log(1.0 * x) / std::log(2.0))

                     unsigned int index = static_cast<unsigned int>
                        (1000.0 * log2(1.75 + i - log2(log2(z))) / log2(max_iterations));
                     #undef log2

                     rgb_t c0 = colormap[0][index];
                     rgb_t c1 = colormap[1][index];
                     rgb_t c2 = colormap[2][index];
                     rgb_t c3 = colormap[3][index];

                     fractal_hsv  .set_pixel(x, y, c0.red, c0.green, c0.blue);
                     fractal_jet  .set_pixel(x, y, c1.red, c1.green, c1.blue);
                     fractal_prism.set_pixel(x, y, c2.red, c2.green, c2.blue);
                     fractal_vga  .set_pixel(x, y, c3.red, c3.green, c3.blue);
                  }

                  break;
               }
            }
         }
      }

      fractal_hsv  .save_image("test20_mandelbrot_set_hsv.bmp"  );
      fractal_jet  .save_image("test20_mandelbrot_set_jet.bmp"  );
      fractal_prism.save_image("test20_mandelbrot_set_prism.bmp");
      fractal_vga  .save_image("test20_mandelbrot_set_vga.bmp"  );
   }

   {
      bitmap_image fractal_hsv  (fractal_width,fractal_height);
      bitmap_image fractal_jet  (fractal_width,fractal_height);
      bitmap_image fractal_prism(fractal_width,fractal_height);
      bitmap_image fractal_vga  (fractal_width,fractal_height);

      fractal_hsv  .clear();
      fractal_jet  .clear();
      fractal_prism.clear();
      fractal_vga  .clear();

      const unsigned int max_iterations = 300;

      const double cr = -0.70000;
      const double ci =  0.27015;

      double prevr, previ;

      for (unsigned int y = 0; y < fractal_height; ++y)
      {
         for (unsigned int x = 0; x < fractal_width; ++x)
         {
            double nextr = 1.5 * (2.0 * x / fractal_width  - 1.0);
            double nexti =       (2.0 * y / fractal_height - 1.0);

            for (unsigned int i = 0; i < max_iterations; i++)
            {
               prevr = nextr;
               previ = nexti;

               nextr =     prevr * prevr - previ * previ + cr;
               nexti = 2 * prevr * previ + ci;

               if (((nextr * nextr) + (nexti * nexti)) > 4)
               {
                  if (max_iterations != i)
                  {
                     unsigned int index = static_cast<int>((1000.0 * i) / max_iterations);

                     rgb_t c0 = colormap[0][index];
                     rgb_t c1 = colormap[1][index];
                     rgb_t c2 = colormap[2][index];
                     rgb_t c3 = colormap[3][index];

                     fractal_hsv  .set_pixel(x, y, c0.red, c0.green, c0.blue);
                     fractal_jet  .set_pixel(x, y, c1.red, c1.green, c1.blue);
                     fractal_prism.set_pixel(x, y, c2.red, c2.green, c2.blue);
                     fractal_vga  .set_pixel(x, y, c3.red, c3.green, c3.blue);
                  }

                  break;
               }
            }
         }
      }

      fractal_hsv  .save_image("test20_julia_set_hsv.bmp"  );
      fractal_jet  .save_image("test20_julia_set_jet.bmp"  );
      fractal_prism.save_image("test20_julia_set_prism.bmp");
      fractal_vga  .save_image("test20_julia_set_vga.bmp"  );
   }
}

int main()
{
   test01();
   test02();
   test03();
   test04();
   test05();
   test06();
   test07();
   test08();
   test09();
   test10();
   test11();
   test12();
   test13();
   test14();
   test15();
   test16();
   test17();
   test18();
   test19();
   test20();
   return 0;
}


/*
   Note: In some of the tests a bitmap image by the name of 'image.bmp'
         is required. If not present the test will fail.
*/
