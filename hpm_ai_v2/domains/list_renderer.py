"""
ListRenderer — renders an HFN node tree to executable Python source.

Ported from experiment_unified_perception_action.py.
"""
from __future__ import annotations

import ast
from typing import List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hpm_ai_v2.domains.base import DomainConfig

import numpy as np


from hpm_ai_v2.utils.base_renderer import Renderer


class ListRenderer(Renderer):
    """Converts an HFN node (or tree) into a Python code string via AST."""

    def __init__(self, config: "DomainConfig"):
        self.config = config

    def _get_concept(self, node: "HFN") -> Optional[str]:
        if node.relation_type == "grounded_op":
            return "GROUNDED_OP"
        for c in self.config.concepts:
            if node.id == f"prior_rule_{c}":
                return c
        s_dim = self.config.S_DIM
        dim = self.config.DIM
        action_vec = node.mu[s_dim: s_dim + dim]
        if np.max(action_vec) > 0.5:
            return self.config.concepts[int(np.argmax(action_vec))]
        return None

    def _get_hfn_leaves(self, node: "HFN") -> List["HFN"]:
        if node is None:
            return []
        if node.relation_type == "grounded_op":
            return [node]
        if node.inputs:
            leaves: List["HFN"] = []
            for n in node.inputs:
                leaves.extend(self._get_hfn_leaves(n))
            return leaves
        children = node.children()
        if not children:
            return [node]
        leaves = []
        for c in children:
            leaves.extend(self._get_hfn_leaves(c))
        return leaves

    def _render_statements(self, node: "HFN") -> List[ast.stmt]:
        if node is None:
            return []
        leaves = self._get_hfn_leaves(node)
        statements: List[ast.stmt] = []
        statements.append(ast.Assign(
            targets=[ast.Name(id='x', ctx=ast.Store())],
            value=ast.Constant(value=0)))
        statements.append(ast.Assign(
            targets=[ast.Name(id='val', ctx=ast.Store())],
            value=ast.Constant(value=0)))
        statements.append(ast.Assign(
            targets=[ast.Name(id='res', ctx=ast.Store())],
            value=ast.Constant(value=None)))

        stack = [statements]
        for leaf in leaves:
            concept = self._get_concept(leaf)
            if not concept:
                # Map common study phase macro IDs to their corresponding concepts
                if "scalar_add1" in leaf.id:
                    concept = "OP_ADD"
                elif "scalar_mul2" in leaf.id:
                    concept = "OP_MUL2"
                else:
                    continue
            if concept == "BLOCK_END":
                if len(stack) > 1:
                    stack.pop()
                continue
            curr_block = stack[-1]
            if concept == "CONST_1":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='x', ctx=ast.Store())],
                    value=ast.Constant(value=1)))
            elif concept == "VAR_INP":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='x', ctx=ast.Store())],
                    value=ast.Name(id='inp', ctx=ast.Load())))
            elif concept == "OP_ADD":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                curr_block.append(ast.AugAssign(
                    target=target, op=ast.Add(),
                    value=ast.Constant(value=1)))
            elif concept == "OP_SUB":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                curr_block.append(ast.AugAssign(
                    target=target, op=ast.Sub(),
                    value=ast.Constant(value=1)))
            elif concept == "GROUNDED_OP":
                delta = leaf.mu[self.config.S_DIM + self.config.DIM + 3]
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                delta_val = int(delta) if float(delta) == int(delta) else float(delta)
                curr_block.append(ast.AugAssign(
                    target=target, op=ast.Add(),
                    value=ast.Constant(value=delta_val)))
            elif concept == "OP_MUL2":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                curr_block.append(ast.AugAssign(
                    target=target, op=ast.Mult(),
                    value=ast.Constant(value=2)))
            elif concept == "OP_SQUARE":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                curr_block.append(ast.Assign(
                    targets=[target],
                    value=ast.BinOp(
                        left=ast.Name(id=target.id, ctx=ast.Load()),
                        op=ast.Pow(),
                        right=ast.Constant(value=2))))
            elif concept == "OP_SQRT":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                curr_block.append(ast.Assign(
                    targets=[target],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='math', ctx=ast.Load()),
                            attr='sqrt', ctx=ast.Load()),
                        args=[ast.Name(id=target.id, ctx=ast.Load())],
                        keywords=[])))
            elif concept == "OP_DIV2":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                target = (ast.Name(id='val', ctx=ast.Store())
                          if len(stack) > 1
                          else ast.Name(id='x', ctx=ast.Store()))
                curr_block.append(ast.AugAssign(
                    target=target, op=ast.Div(),
                    value=ast.Constant(value=2)))
            elif concept == "LIST_INIT":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='res', ctx=ast.Store())],
                    value=ast.List(elts=[], ctx=ast.Load())))
            elif concept == "FOR_LOOP":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                list_call = ast.Call(
                    func=ast.Name(id='list', ctx=ast.Load()),
                    args=[ast.Name(id='x', ctx=ast.Load())], keywords=[])
                for_node = ast.For(
                    target=ast.Name(id='item', ctx=ast.Store()),
                    iter=list_call, body=[ast.Pass()], orelse=[])
                curr_block.append(for_node)
                stack.append(for_node.body)
            elif concept == "ITEM_ACCESS":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='val', ctx=ast.Store())],
                    value=ast.Name(id='item', ctx=ast.Load())))
            elif concept == "LIST_APPEND":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                append_call = ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='res', ctx=ast.Load()),
                        attr='append', ctx=ast.Load()),
                    args=[ast.Name(id='val', ctx=ast.Load())], keywords=[]))
                curr_block.append(append_call)
            elif concept == "MAP_START":
                # Composite: VAR_INP + LIST_INIT + FOR_LOOP + ITEM_ACCESS
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                # 1. x = inp
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='x', ctx=ast.Store())],
                    value=ast.Name(id='inp', ctx=ast.Load())))
                # 2. res = []
                curr_block.append(ast.Assign(
                    targets=[ast.Name(id='res', ctx=ast.Store())],
                    value=ast.List(elts=[], ctx=ast.Load())))
                # 3. for item in list(x): val = item
                for_node = ast.For(
                    target=ast.Name(id='item', ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id='list', ctx=ast.Load()),
                        args=[ast.Name(id='x', ctx=ast.Load())], keywords=[]),
                    body=[ast.Assign(
                        targets=[ast.Name(id='val', ctx=ast.Store())],
                        value=ast.Name(id='item', ctx=ast.Load()))],
                    orelse=[])
                curr_block.append(for_node)
                stack.append(for_node.body)
            elif concept == "MAP_END":
                # Composite: LIST_APPEND + BLOCK_END
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                # 1. res.append(val)
                curr_block.append(ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='res', ctx=ast.Load()),
                        attr='append', ctx=ast.Load()),
                    args=[ast.Name(id='val', ctx=ast.Load())], keywords=[])))
                # 2. BLOCK_END
                if len(stack) > 1:
                    stack.pop()
            elif concept == "COND_IS_EVEN":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                if_node = ast.If(
                    test=ast.Compare(
                        left=ast.BinOp(
                            left=ast.Name(id='val', ctx=ast.Load()),
                            op=ast.Mod(), right=ast.Constant(value=2)),
                        ops=[ast.Eq()], comparators=[ast.Constant(value=0)]),
                    body=[ast.Pass()], orelse=[])
                curr_block.append(if_node)
                stack.append(if_node.body)
            elif concept == "COND_IS_POSITIVE":
                if curr_block and isinstance(curr_block[-1], ast.Pass):
                    curr_block.pop()
                if_node = ast.If(
                    test=ast.Compare(
                        left=ast.Name(id='val', ctx=ast.Load()),
                        ops=[ast.Gt()], comparators=[ast.Constant(value=0)]),
                    body=[ast.Pass()], orelse=[])
                curr_block.append(if_node)
                stack.append(if_node.body)
            elif concept == "RETURN":
                ret_val = ast.IfExp(
                    test=ast.Compare(
                        left=ast.Name(id='res', ctx=ast.Load()),
                        ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
                    body=ast.Name(id='res', ctx=ast.Load()),
                    orelse=ast.Name(id='x', ctx=ast.Load()))
                statements.append(ast.Return(value=ret_val))
                break

        if not any(isinstance(s, ast.Return) for s in statements):
            ret_val = ast.IfExp(
                test=ast.Compare(
                    left=ast.Name(id='res', ctx=ast.Load()),
                    ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
                body=ast.Name(id='res', ctx=ast.Load()),
                orelse=ast.Name(id='x', ctx=ast.Load()))
            statements.append(ast.Return(value=ret_val))
        return statements

    def render(self, node: "HFN") -> str:
        if node is None:
            return ""
        # Code macros carry their source directly — skip AST construction.
        if getattr(node, '_code', None):
            return node._code

        statements = self._render_statements(node)
        module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(module)
        return "import math\n" + ast.unparse(module)

    def render_function(self, node: "HFN", func_name: str = "macro_func") -> str:
        """Render a macro as a standalone Python function definition."""
        if getattr(node, '_code', None):
            code = node._code
            # If it already looks like a function with this name, return it.
            if f"def {func_name}(" in code:
                return code
            # Otherwise, wrap it or parse and rename. For robustness, we wrap.
            indented = code.replace('\n', '\n    ')
            return f"import math\ndef {func_name}(inp):\n    {indented}"

        statements = self._render_statements(node)
        func_def = ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg='inp')],
                kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=statements,
            decorator_list=[])
        module = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(module)
        return "import math\n" + ast.unparse(module)
