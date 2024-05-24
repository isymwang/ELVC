# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-src"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-build"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/tmp"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src"
  "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "G:/wangyiming/optical_flow_guided_feature_compression_baseline/fvc_net/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp${cfgdir}") # cfgdir has leading slash
endif()
