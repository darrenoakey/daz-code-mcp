#!/usr/bin/env python3
"""
Debug script to test tree-sitter imports and find correct function names
"""

print("Testing tree-sitter imports and finding correct function names...")

# Test basic tree-sitter
try:
    from tree_sitter import Parser

    print("✓ tree-sitter Parser imported successfully")
except ImportError as e:
    print(f"✗ Failed to import tree-sitter Parser: {e}")
    exit(1)

# Test each language individually and inspect available attributes
language_modules = [
    ("python", "tree_sitter_python"),
    ("javascript", "tree_sitter_javascript"),
    ("typescript", "tree_sitter_typescript"),
    ("java", "tree_sitter_java"),
    ("cpp", "tree_sitter_cpp"),
    ("go", "tree_sitter_go"),
    ("rust", "tree_sitter_rust"),
]

working_languages = {}

for lang_name, module_name in language_modules:
    print(f"\n--- Testing {module_name} ---")
    try:
        module = __import__(module_name)
        print(f"✓ {module_name} imported successfully")

        # List all attributes
        attrs = [attr for attr in dir(module) if not attr.startswith("_")]
        print(f"Available attributes: {attrs}")

        # Try common function name patterns
        possible_names = [
            f"language_{lang_name}",
            f"language",
            f"{lang_name}",
            f"LANGUAGE",
            f"Language",
        ]

        language_obj = None
        working_function = None

        for func_name in possible_names:
            if hasattr(module, func_name):
                try:
                    func = getattr(module, func_name)
                    if callable(func):
                        language_obj = func()
                        working_function = func_name
                        print(
                            f"✓ {func_name}() works and returns: {type(language_obj)}"
                        )
                        break
                    else:
                        print(f"- {func_name} exists but is not callable: {type(func)}")
                        # Might be the language object itself
                        language_obj = func
                        working_function = func_name
                        print(
                            f"✓ {func_name} is the language object: {type(language_obj)}"
                        )
                        break
                except Exception as e:
                    print(f"- {func_name}() exists but failed: {e}")

        if language_obj and working_function:
            working_languages[lang_name] = (module_name, working_function, language_obj)
            print(f"✓ WORKING: {module_name}.{working_function}")
        else:
            print(f"✗ No working language function found in {module_name}")

    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
    except Exception as e:
        print(f"✗ Error with {module_name}: {e}")

print(f"\n=== SUMMARY ===")
print(f"Working languages: {len(working_languages)}")
for lang, (module, func, obj) in working_languages.items():
    print(f"  {lang}: {module}.{func} -> {type(obj)}")

# Test creating a parser with one of the languages
if working_languages:
    try:
        parser = Parser()
        first_lang = list(working_languages.values())[0][2]  # Get the language object
        parser.set_language(first_lang)
        print("\n✓ Parser creation and language setting works")
    except Exception as e:
        print(f"\n✗ Parser setup failed: {e}")
else:
    print("\n✗ No working languages found")
