# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-src"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-build"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/tmp"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp${cfgdir}") # cfgdir has leading slash
endif()
