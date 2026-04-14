"""
ImageCodeGenerator — Generates PIL AST nodes for image operations.
"""
from __future__ import annotations
import ast
from typing import Any, Dict

class ImageCodeGenerator:
    """
    Translates image domain concepts into PIL-based Python AST.
    """
    def generate(self, concept: str, context: Dict[str, Any]) -> ast.AST:
        var = context.get("var", "img")
        
        if concept == "ROTATE_90":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="rotate", ctx=ast.Load()),
                    args=[ast.Constant(value=-90)],
                    keywords=[]
                )
            )
        elif concept == "ROTATE_180":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="rotate", ctx=ast.Load()),
                    args=[ast.Constant(value=-180)],
                    keywords=[]
                )
            )
        elif concept == "ROTATE_270":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="rotate", ctx=ast.Load()),
                    args=[ast.Constant(value=-270)],
                    keywords=[]
                )
            )
        elif concept == "FLIP_H":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="transpose", ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id="Image", ctx=ast.Load()), attr="FLIP_LEFT_RIGHT", ctx=ast.Load())],
                    keywords=[]
                )
            )
        elif concept == "FLIP_V":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="transpose", ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id="Image", ctx=ast.Load()), attr="FLIP_TOP_BOTTOM", ctx=ast.Load())],
                    keywords=[]
                )
            )
        elif concept == "BLUR":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="filter", ctx=ast.Load()),
                    args=[ast.Call(
                        func=ast.Attribute(value=ast.Name(id="ImageFilter", ctx=ast.Load()), attr="GaussianBlur", ctx=ast.Load()),
                        args=[ast.Constant(value=1)],
                        keywords=[]
                    )],
                    keywords=[]
                )
            )
        elif concept == "BRIGHTNESS_UP":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id="ImageEnhance", ctx=ast.Load()), attr="Brightness", ctx=ast.Load()),
                            args=[ast.Name(id=var, ctx=ast.Load())],
                            keywords=[]
                        ),
                        attr="enhance",
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=1.5)],
                    keywords=[]
                )
            )
        elif concept == "BRIGHTNESS_DOWN":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id="ImageEnhance", ctx=ast.Load()), attr="Brightness", ctx=ast.Load()),
                            args=[ast.Name(id=var, ctx=ast.Load())],
                            keywords=[]
                        ),
                        attr="enhance",
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=0.5)],
                    keywords=[]
                )
            )
        elif concept == "EDGE_DETECT":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=var, ctx=ast.Load()), attr="filter", ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id="ImageFilter", ctx=ast.Load()), attr="FIND_EDGES", ctx=ast.Load())],
                    keywords=[]
                )
            )
        elif concept == "CONTRAST_UP":
            return ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id="ImageEnhance", ctx=ast.Load()), attr="Contrast", ctx=ast.Load()),
                            args=[ast.Name(id=var, ctx=ast.Load())],
 # Wait, typo in design? "Load" instead of "ast.Load"
                            keywords=[]
                        ),
                        attr="enhance",
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=1.5)],
                    keywords=[]
                )
            )
        
        # Default to identity
        return ast.Pass()
