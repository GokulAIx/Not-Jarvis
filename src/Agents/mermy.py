from .agent import workflow

w=workflow.compile()

png_bytes = w.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved graph.png")
