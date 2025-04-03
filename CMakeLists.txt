cmake_minimum_required(VERSION 3.10)
project(Module-4 CXX)

# Задаём стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Флаги оптимизации для разных конфигураций
# if(MSVC)
#     # Для Visual Studio
#     set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DNDEBUG")
#     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /fp:fast /Zi /DNDEBUG")
#     set(CMAKE_CXX_FLAGS_MINSIZEREL "/O1 /fp:fast /DNDEBUG")
# else()
#     # Для GCC/Clang
#     set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
#     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -g -DNDEBUG")
#     set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -march=native -DNDEBUG")
# endif()

# путь к Eigen через vcpkg
find_package(Eigen3 REQUIRED)

# Исходные файлы проекта
set(SOURCES module4.cpp)
set(HEADERS ModelData.h)

# Основная библиотека
add_library(module4 STATIC module4.cpp ModelData.h)
target_include_directories(module4 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}  # Добавлено
)
# Применить оптимизацию к библиотеке
# target_compile_options(module4 PRIVATE 
#     $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}>
#     $<$<CONFIG:RelWithDebInfo>:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}>
#     $<$<CONFIG:MinSizeRel>:${CMAKE_CXX_FLAGS_MINSIZEREL}>
# )

# Настройка GoogleTest через vcpkg
enable_testing()
find_package(GTest CONFIG REQUIRED)

# тестовый исполняемый файл для функций из module4
add_executable(test_module4 test_module4.cpp)
target_link_libraries(test_module4 PRIVATE module4 GTest::gtest GTest::gtest_main)

add_test(NAME Module4Test COMMAND test_module4)

find_package(Doxygen)
if (DOXYGEN_FOUND)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        add_custom_target(doc ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Генерация документации Doxygen"
            VERBATIM)
    else()
        message(WARNING "Файл Doxyfile не найден в ${CMAKE_CURRENT_SOURCE_DIR}. Цель 'doc' не создана.")
    endif()
else()
    message(WARNING "Doxygen не найден. Документация не будет сгенерирована.")
endif()