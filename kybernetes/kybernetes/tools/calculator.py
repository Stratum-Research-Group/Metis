"""Safe, deterministic AST-based calculator tool."""

import ast
import math
import operator
from typing import Any, Dict

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition, ToolResult


class SafeCalculator:
    """
    Secure mathematical evaluator using Python's AST parser.
    Zero use of eval() or exec(); prevents code injection and arbitrary execution.
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
        "exp": math.exp,
    }

    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            version="1.0.0",
            description="Safely evaluate mathematical expressions deterministically.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical formula to evaluate, e.g. 'sqrt(144) + 12 * 3'",
                    }
                },
                "required": ["expression"],
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "number"},
                    "expression": {"type": "string"},
                },
            },
            declared_capabilities=["compute"],
            side_effect_class=ToolSideEffectClass.READ_ONLY,
            timeout_ms=3000,
            idempotent=True,
        )

    def evaluate(self, expression: str) -> float:
        """Parse and evaluate math expression AST safely."""
        expr_clean = expression.strip()
        if not expr_clean:
            raise ValueError("Empty mathematical expression")

        try:
            tree = ast.parse(expr_clean, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {e}") from e

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero in mathematical expression")
            if op_type == ast.Pow and right > 1000:
                raise ValueError("Exponent too large (max 1000)")
            return float(self.OPERATORS[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return float(self.OPERATORS[op_type](operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Arbitrary callable invocations are prohibited")
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Disallowed math function: '{func_name}'")
            args = [self._eval_node(a) for a in node.args]
            if func_name == "round" and len(args) == 2:
                args[1] = int(args[1])
            return float(self.ALLOWED_FUNCTIONS[func_name](*args))

        if isinstance(node, ast.Name):
            name = node.id
            if name in self.ALLOWED_CONSTANTS:
                return float(self.ALLOWED_CONSTANTS[name])
            raise ValueError(f"Disallowed identifier/variable: '{name}'")

        raise ValueError(f"Disallowed AST construct: {type(node).__name__}")
