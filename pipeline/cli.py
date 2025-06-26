from pipeline.dag import build_graph

def main():
    graph = build_graph()

    while True:
        txt = input("\nEnter text (or 'exit'): ")
        if txt.strip().lower() == "exit":
            break

        res = graph.invoke({"input_text": txt})  # ✅ FIXED: use 'input_text'
        print("\nResult:")
        print(res)

if __name__ == "__main__":
    main()