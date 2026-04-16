from __future__ import annotations
import ast
from typing import List, Optional, Any, TYPE_CHECKING
import numpy as np

from hpm_ai_v2.utils.base_renderer import Renderer
if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hpm_ai_v2.domains.video_domain import VideoDomainConfig

class VideoRenderer(Renderer):
    """
    Converts HFN node trees into Python code for video transformations.
    Uses cv2 and numpy for efficient frame processing.
    """
    def __init__(self, config: VideoDomainConfig):
        self.config = config

    def _get_hfn_leaves(self, node: HFN) -> List[HFN]:
        """Extract leaves from a macro or sequence tree."""
        if node is None:
            return []
        if node.relation_type == "grounded_op":
            return [node]
        if node.inputs:
            leaves: List[HFN] = []
            for n in node.inputs:
                leaves.extend(self._get_hfn_leaves(n))
            return leaves
        
        children = node.children()
        if not children:
            return [node]
        leaves = []
        for child in children:
            leaves.extend(self._get_hfn_leaves(child))
        return leaves

    def _get_concept(self, node: HFN) -> Optional[str]:
        """Reverse map a node back to a concept string."""
        if node.id and node.id.startswith("prior_rule_"):
            return node.id.replace("prior_rule_", "")
        # Geometric lookup: find the concept index with the highest weight in mu
        s_dim = self.config.S_DIM
        dim = self.config.DIM
        concept_vec = node.mu[s_dim : s_dim + dim]
        if np.max(concept_vec) > 1.0:
            idx = np.argmax(concept_vec)
            return self.config.concepts[idx]
        return None

    def _render_statements(self, node: HFN) -> List[ast.stmt]:
        leaves = self._get_hfn_leaves(node)
        statements: List[ast.stmt] = []
        
        # Initial variable setup
        statements.append(ast.Assign(
            targets=[ast.Name(id='x', ctx=ast.Store())],
            value=ast.Constant(value=None)))
        statements.append(ast.Assign(
            targets=[ast.Name(id='val', ctx=ast.Store())],
            value=ast.Constant(value=None)))
        statements.append(ast.Assign(
            targets=[ast.Name(id='res', ctx=ast.Store())],
            value=ast.Constant(value=None)))

        stack = [statements]
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept:
                # Handle study phase macros
                if "scalar_add1" in leaf.id or "brightness_up" in leaf.id or "macro_bright_up" in leaf.id:
                    concept = "BRIGHTNESS_UP"
                elif "scalar_mul2" in leaf.id or "brightness_down" in leaf.id or "macro_bright_down" in leaf.id:
                    concept = "BRIGHTNESS_DOWN"
                elif "macro_rotate" in leaf.id:
                    concept = "ROTATE_90"
                elif "macro_flip_h" in leaf.id:
                    concept = "FLIP_H"
                else:
                    continue

            if concept == "BLOCK_END":
                if len(stack) > 1:
                    stack.pop()
                continue
            
            curr_block = stack[-1]
            if concept == "VAR_INP":
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='x', ctx=ast.Store())],
                    value=ast.Name(id='inp', ctx=ast.Load())))
            elif concept == "LIST_INIT":
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='res', ctx=ast.Store())],
                    value=ast.List(elts=[], ctx=ast.Load())))
            elif concept == "FOR_EACH_FRAME":
                # for frame in list(x):
                for_node = ast.For(
                    target=ast.Name(id='item', ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id='list', ctx=ast.Load()),
                        args=[ast.Name(id='x', ctx=ast.Load())], keywords=[]),
                    body=[ast.Pass()],
                    orelse=[])
                curr_block.append(for_node)
                stack.append(for_node.body)
            elif concept == "ITEM_ACCESS":
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Name(id='item', ctx=ast.Load())))
            elif concept == "FRAME_APPEND":
                # res.append(val)
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='res', ctx=ast.Load()),
                        attr='append', ctx=ast.Load()),
                    args=[ast.Name(id='val', ctx=ast.Load())], keywords=[])))
            elif concept == "ROTATE_90":
                # val = cv2.rotate(val, cv2.ROTATE_90_CLOCKWISE)
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='cv2', ctx=ast.Load()), attr='rotate', ctx=ast.Load()),
                        args=[ast.Name(id='val', ctx=ast.Load()), ast.Attribute(value=ast.Name(id='cv2', ctx=ast.Load()), attr='ROTATE_90_CLOCKWISE', ctx=ast.Load())], keywords=[])))
            elif concept == "FLIP_H":
                # val = cv2.flip(val, 1)
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='cv2', ctx=ast.Load()), attr='flip', ctx=ast.Load()),
                        args=[ast.Name(id='val', ctx=ast.Load()), ast.Constant(value=1)], keywords=[])))
            elif concept == "FLIP_V":
                # val = cv2.flip(val, 0)
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='cv2', ctx=ast.Load()), attr='flip', ctx=ast.Load()),
                        args=[ast.Name(id='val', ctx=ast.Load()), ast.Constant(value=0)], keywords=[])))
            elif concept == "BRIGHTNESS_UP":
                # val = np.clip(val.astype(float) + 30, 0, 255).astype(np.uint8)
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='clip', ctx=ast.Load()),
                        args=[
                            ast.BinOp(left=ast.Call(func=ast.Attribute(value=ast.Name(id='val', ctx=ast.Load()), attr='astype', ctx=ast.Load()), args=[ast.Name(id='float', ctx=ast.Load())], keywords=[]), op=ast.Add(), right=ast.Constant(value=30)),
                            ast.Constant(value=0), ast.Constant(value=255)
                        ], keywords=[])))
            elif concept == "BRIGHTNESS_DOWN":
                # val = np.clip(val.astype(float) - 30, 0, 255).astype(np.uint8)
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='clip', ctx=ast.Load()),
                        args=[
                            ast.BinOp(left=ast.Call(func=ast.Attribute(value=ast.Name(id='val', ctx=ast.Load()), attr='astype', ctx=ast.Load()), args=[ast.Name(id='float', ctx=ast.Load())], keywords=[]), op=ast.Sub(), right=ast.Constant(value=30)),
                            ast.Constant(value=0), ast.Constant(value=255)
                        ], keywords=[])))
            elif concept == "MAP_START":
                # 1. x = inp; res = []; for item in list(x): val = item
                curr_block.append(ast.Assign(targets=[ast.Name(id='x', ctx=ast.Store())], value=ast.Name(id='inp', ctx=ast.Load())))
                curr_block.append(ast.Assign(targets=[ast.Name(id='res', ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
                for_node = ast.For(
                    target=ast.Name(id='item', ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id='list', ctx=ast.Load()), args=[ast.Name(id='x', ctx=ast.Load())], keywords=[]),
                    body=[ast.Assign(targets=[ast.Name(id='val', ctx=ast.Store())], value=ast.Name(id='item', ctx=ast.Load()))],
                    orelse=[])
                curr_block.append(for_node)
                stack.append(for_node.body)
            elif concept == "MAP_END":
                # 1. res.append(val); BLOCK_END
                if curr_block and isinstance(curr_block[-1], ast.Pass): curr_block.pop()
                curr_block.append(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='res', ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Name(id='val', ctx=ast.Load())], keywords=[])))
                if len(stack) > 1: stack.pop()
            elif concept == "RETURN":
                curr_block.append(ast.Return(value=ast.Name(id='res', ctx=ast.Load())))

        # If no explicit return, add it
        if not any(isinstance(s, ast.Return) for s in statements):
            statements.append(ast.Return(value=ast.Name(id='res', ctx=ast.Load())))
            
        return statements

    def render(self, node: HFN) -> str:
        if node is None: return ""
        if getattr(node, '_code', None): return node._code
        statements = self._render_statements(node)
        module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(module)
        return "import cv2\nimport numpy as np\n" + ast.unparse(module)

    def render_function(self, node: HFN, func_name: str = "video_macro") -> str:
        if getattr(node, '_code', None):
            code = node._code
            if f"def {func_name}(" in code: return code
            indented = code.replace('\n', '\n    ')
            return f"import cv2\nimport numpy as np\ndef {func_name}(inp):\n    {indented}"
        statements = self._render_statements(node)
        func_def = ast.FunctionDef(name=func_name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='inp')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=statements, decorator_list=[])
        module = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(module)
        return "import cv2\nimport numpy as np\n" + ast.unparse(module)
