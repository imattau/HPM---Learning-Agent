"""
ASTRenderer — renders an HFN node tree to executable Python source.

Ported from experiment_unified_perception_action.py.
"""
from __future__ import annotations

import ast
from typing import List, Optional, TYPE_CHECKING

from hpm_ai_v2.utils.state import CONCEPTS, S_DIM, DIM

if TYPE_CHECKING:
    from hfn.hfn import HFN

import numpy as np


class ASTRenderer:
    """Converts an HFN node (or tree) into a Python code string via AST."""

    def _get_concept(self, node: "HFN") -> Optional[str]:
        if node.relation_type == "grounded_op":
            return "GROUNDED_OP"
        for c in CONCEPTS:
            if node.id == f"prior_rule_{c}":
                return c
        action_vec = node.mu[S_DIM: S_DIM + DIM]
        if np.max(action_vec) > 0.5:
            return CONCEPTS[int(np.argmax(action_vec))]
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

    def render(self, node: "HFN") -> str:
        if node is None:
            return ""
        leaves = self._get_hfn_leaves(node)
        statements: list = []
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
                continue
            if concept in {"OP_ADD", "OP_SUB"}:
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
            elif concept == "GROUNDED_OP":
                delta = leaf.mu[S_DIM + DIM + 3]
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

        module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)
