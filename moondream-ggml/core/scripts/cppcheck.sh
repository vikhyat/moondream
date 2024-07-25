cppcheck --std=c++11 \
    --language=c++ \
    --enable=all \
    --check-level=exhaustive \
    --suppress=missingIncludeSystem \
    --suppress=missingInclude \
    ./src/moondream.cpp
