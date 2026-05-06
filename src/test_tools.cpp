#include <iostream>
#include <vector>
#include <memory>

// ── clang-tidy tests ─────────────────────────────────────────────────────────

// TEST 1: modernize-use-nullptr
// clang-tidy should flag NULL and suggest nullptr
void test_null() {
    int* p = NULL;          // should warn: use nullptr
    char* s = NULL;         // should warn: use nullptr
}

// TEST 2: modernize-use-auto
// clang-tidy should suggest auto for obvious types
void test_auto() {
    std::vector<int> v = {1, 2, 3};
    std::vector<int>::iterator it = v.begin(); // should suggest auto
}

// TEST 3: modernize-use-override
// clang-tidy should flag missing override
struct Base {
    virtual void foo() {}
    virtual ~Base() {}
};
struct Derived : Base {
    void foo() {}   // should warn: add override
};

// TEST 4: unused variable (compiler warning -Wall)
// should show a warning squiggle
void test_unused() {
    int unused_var = 42;    // should warn: unused variable
}

// TEST 5: modernize-use-using (typedef → using)
typedef int MyInt;          // should suggest: using MyInt = int;

// TEST 6: readability — magic numbers (if you have that check enabled)
void test_magic() {
    int x = 42 * 1337;     // magic numbers
}

// ── clang-format tests ───────────────────────────────────────────────────────

// TEST 7: badly formatted code
// save the file then run :lua vim.lsp.buf.format()
// or in normal mode: <leader>cf
// it should reformat to match your .clang-format style
void badly_formatted(int x,int y,int z){
int result=x+y+z;
    return   ;
}

// TEST 8: include order (clang-format sorts includes)
// should be sorted after format

// ── clangd LSP tests ─────────────────────────────────────────────────────────

// TEST 9: go to definition — put cursor on vector and press gd
// should jump to the STL header

// TEST 10: hover — put cursor on std::vector and press K
// should show the type signature

// TEST 11: autocomplete — type std:: and trigger completion
// should show all std:: symbols

// TEST 12: type error — clangd should show red squiggle immediately
void test_type_error() {
    int x = "hello";       // type error: clangd should catch this
}

// TEST 13: missing return
int test_missing_return() {
                           // should warn: control reaches end of non-void function
}

int main() {
    test_null();
    test_unused();
    return 0;
}
