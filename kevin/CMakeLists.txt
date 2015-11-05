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

set(DATFILETYPES
  "enuc"
  "geom"
  "s"
  "t"
  "v"
  "eri"
  "mux"
  "muy"
  "muz"
  )
set(MOLECULES
  "h2o_sto3g"
  )

foreach(imolecule
    ${MOLECULES}
    )
    foreach(datfile
        ${DATAFILETYPES}
        )
        add_custom_command(
        OUTPUT
        "${imolecule}_${datfile}.dat"
        COMMAND "${CMAKE_COMMAND}" -E copy
        "${programming_party_SOURCE_DIR}/project3/${imolecule}_${datfile}.dat"
        "${programming_party_BINARY_DIR}/${imolecule}_${datfile}.dat"
        DEPENDS
        "${programming_party_SOURCE_DIR}/project3/${imolecule}_${datfile}.dat"
        )
      endforeach(datfile)
 endforeach(imolecule)


 #add_custom_target(inp_p3_h2o_sto3g
 # DEPENDS
 #   foreach(imolecule
 #     ${MOLECULES}
 #     )
 #     foreach(datfile
 #        ${DATAFILETYPES}
 #        )
 #        "${programming_party_BINARY_DIR}/${imolecule}_${datfile}.dat"
 #      endforeach(datfile)
 #     endforeach(imolecule)
 #   )


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

