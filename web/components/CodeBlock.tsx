"use client";

import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";

export function CodeBlock({ code, language = "python" }: { code: string; language?: string }) {
  return (
    <div className="rounded-xl overflow-hidden text-sm" style={{ border: "1px solid var(--border)" }}>
      <SyntaxHighlighter
        language={language}
        style={atomOneDark}
        customStyle={{
          margin: 0,
          padding: "1rem",
          background: "#0a0a0a",
          fontSize: 13,
          lineHeight: 1.55,
        }}
        wrapLongLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
