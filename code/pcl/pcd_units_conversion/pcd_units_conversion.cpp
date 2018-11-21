#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);


  if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (argv[1], *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read pcd file \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from " << argv[1]
            << std::endl;

  pcl::PointCloud<pcl::PointXYZRGBA> new_cloud;

  new_cloud.width = cloud->width;
  new_cloud.height = cloud->height;
  new_cloud.is_dense = false;
  new_cloud.points.resize (new_cloud.width * new_cloud.height);

  for (size_t i = 0; i < new_cloud.points.size (); ++i)
  {
    new_cloud.points[i].x = cloud->points[i].x/1000;
    new_cloud.points[i].y = cloud->points[i].y/1000;
    new_cloud.points[i].z = cloud->points[i].z/1000;
    new_cloud.points[i].rgb = cloud->points[i].rgb;
  }

  pcl::io::savePCDFileASCII (argv[2], new_cloud);
  std::cerr << "Saved " << new_cloud.points.size () << " scaled data points to " << argv[2] << std::endl;

  // for (size_t i = 0; i < new_cloud.points.size (); ++i)
  //   std::cerr << "    " << new_cloud.points[i].x << " " << new_cloud.points[i].y << " " << new_cloud.points[i].z << std::endl;

  return (0);
}
