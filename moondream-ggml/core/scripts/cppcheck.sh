cppcheck --std=c++11 \
    --language=c++ \
    --check-level=exhaustive \
    --suppress=missingIncludeSystem \
    src/moondream.cpp
