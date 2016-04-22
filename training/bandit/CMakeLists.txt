INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../utils)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../mteval)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../decoder)

set(bandit_mt_grad_SRCS
    bandit_mt_grad.cc
    bandit_mt.h)
add_executable(bandit_mt_grad ${bandit_mt_grad_SRCS})
target_link_libraries(bandit_mt_grad libcdec ksearch mteval utils klm klm_util klm_util_double ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LIBLZMA_LIBRARIES} ${LIBDL_LIBRARIES})



set(bandit_mt_SRCS
    bandit_mt.cc
    bandit_mt.h)
add_executable(bandit_mt ${bandit_mt_SRCS})
target_link_libraries(bandit_mt libcdec ksearch mteval utils klm klm_util klm_util_double ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LIBLZMA_LIBRARIES} ${LIBDL_LIBRARIES})

set(collect_weights_SRCS
    collect_weights.cc
    bandit_mt.h)
add_executable(collect_weights ${collect_weights_SRCS})
target_link_libraries(collect_weights libcdec ksearch mteval utils klm klm_util klm_util_double ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LIBLZMA_LIBRARIES} ${LIBDL_LIBRARIES})
