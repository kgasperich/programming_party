cmake_minimum_required(VERSION 2.8)
project(programming_party CXX)

set(CMAKE_BUILD_TYPE Release)

find_package(Armadillo)
if (ARMADILLO_FOUND)
  include_directories("${ARMADILLO_INCLUDE_DIRS}")
endif (ARMADILLO_FOUND)

add_library(molecule 
  diag.cpp 
  utils.hpp 
  utils.cpp 
  molecule.hpp 
  molecule.cpp
  )

foreach(datfile 
    h2o_sto3g_enuc
    h2o_sto3g_geom
    h2o_sto3g_s
    h2o_sto3g_t
    h2o_sto3g_v
    h2o_sto3g_eri
    h2o_sto3g_mux
    h2o_sto3g_muy
    h2o_sto3g_muz
    )
  add_custom_command(
    OUTPUT
    "${datfile}.dat"
    COMMAND "${CMAKE_COMMAND}" -E copy
    "${programming_party_SOURCE_DIR}/project3/${datfile}.dat"
    "${programming_party_BINARY_DIR}/${datfile}.dat"
    DEPENDS
    "${programming_party_SOURCE_DIR}/project3/${datfile}.dat"
    )
endforeach(datfile)

#add_custom_command(
#  OUTPUT 
#  "h2o_sto3g_enuc.dat"
#  "h2o_sto3g_s.dat"
#  "h2o_sto3g_t.dat"
#  "h2o_sto3g_v.dat"
#  "h2o_sto3g_eri.dat"
#  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_enuc.dat" "${programming_party_BINARY_DIR}/h2o_sto3g_enuc.dat"
#  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_s.dat" "${programming_party_BINARY_DIR}/h2o_sto3g_s.dat"
#  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_t.dat" "${programming_party_BINARY_DIR}/h2o_sto3g_t.dat"
#  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_v.dat" "${programming_party_BINARY_DIR}/h2o_sto3g_v.dat"
#  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_eri.dat" "${programming_party_BINARY_DIR}/h2o_sto3g_eri.dat"
#  DEPENDS
#  "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_enuc.dat"
#  "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_s.dat"
#  "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_t.dat"
#  "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_v.dat"
#  "${programming_party_SOURCE_DIR}/project3/h2o_sto3g_eri.dat"
#  )

add_custom_target(inp_p3_h2o_sto3g 
  DEPENDS 
  "${programming_party_BINARY_DIR}/h2o_sto3g_enuc.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_geom.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_mux.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_muy.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_muz.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_s.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_t.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_v.dat"
  "${programming_party_BINARY_DIR}/h2o_sto3g_eri.dat"
  )

if (ARMADILLO_FOUND)
  add_executable(project3 project3/project3.cpp)
  add_dependencies(project3 inp_p3_h2o_sto3g)
  target_link_libraries(project3
    molecule
    "${ARMADILLO_LIBRARIES}"
    )
endif (ARMADILLO_FOUND)

