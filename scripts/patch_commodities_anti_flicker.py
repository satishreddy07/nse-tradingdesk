#!/usr/bin/env python3
"""
Patch updateCommoditiesLiveStream in web/templates/index.html to eliminate
layout thrashing and cell flickering.
"""

def main():
    path = "web/templates/index.html"
    with open(path, "rb") as f:
        content_bytes = f.read()

    crlf = b"\r\n" in content_bytes
    sep = "\r\n" if crlf else "\n"
    text = content_bytes.decode("utf-8")

    old_comm_card_update = (
        "        // Update Card LTP with visual flash\n"
        "        const cardLtpEl = document.getElementById(`comm-card-ltp-${baseSym}`);\n"
        "        if (cardLtpEl) {\n"
        "          const oldLtp = parseFloat(cardLtpEl.getAttribute('data-prev-ltp') || '0');\n"
        "          cardLtpEl.innerText = `₹${newLtp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;\n"
        "          cardLtpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "          if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.01) {\n"
        "            const flashCls = newLtp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "            cardLtpEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "            setTimeout(() => cardLtpEl.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "          }\n"
        "        }\n"
        "\n"
        "        // Update Card Change Badge\n"
        "        const cardChgEl = document.getElementById(`comm-card-chg-${baseSym}`);\n"
        "        if (cardChgEl) {\n"
        "          cardChgEl.className = `inline-flex items-center gap-1 px-2.5 py-1 rounded-xl text-xs font-mono font-bold border ${chgCls}`;\n"
        "          cardChgEl.innerHTML = `<span>${sign}${chgPct.toFixed(2)}%</span><span class=\"text-[10px] opacity-80\">(${sign}${chg.toFixed(2)})</span>`;\n"
        "        }\n"
        "\n"
        "        // Update Table Row LTP with visual flash\n"
        "        const rowLtpEl = document.getElementById(`comm-row-ltp-${baseSym}`);\n"
        "        if (rowLtpEl) {\n"
        "          const oldLtp = parseFloat(rowLtpEl.getAttribute('data-prev-ltp') || '0');\n"
        "          rowLtpEl.innerText = `₹${newLtp.toFixed(2)}`;\n"
        "          rowLtpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "          if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.01) {\n"
        "            const flashCls = newLtp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "            rowLtpEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "            setTimeout(() => rowLtpEl.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "          }\n"
        "        }\n"
        "\n"
        "        // Update Table Row Change Cell\n"
        "        const rowChgEl = document.getElementById(`comm-row-chg-${baseSym}`);\n"
        "        if (rowChgEl) {\n"
        "          rowChgEl.className = `py-2.5 px-3 text-right ${isUp ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}`;\n"
        "          rowChgEl.innerText = `${sign}${chgPct.toFixed(2)}%`;\n"
        "        }"
    ).replace("\n", sep)

    new_comm_card_update = (
        "        // Update Card LTP with visual flash (equality guarded)\n"
        "        const cardLtpEl = document.getElementById(`comm-card-ltp-${baseSym}`);\n"
        "        if (cardLtpEl) {\n"
        "          const oldLtp = parseFloat(cardLtpEl.getAttribute('data-prev-ltp') || '0');\n"
        "          if (oldLtp === 0 || Math.abs(newLtp - oldLtp) >= 0.01) {\n"
        "            setSafeText(cardLtpEl, `₹${newLtp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);\n"
        "            cardLtpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "            if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.01) {\n"
        "              const flashCls = newLtp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "              cardLtpEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "              setTimeout(() => cardLtpEl.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "            }\n"
        "          }\n"
        "        }\n"
        "\n"
        "        // Update Card Change Badge\n"
        "        const cardChgEl = document.getElementById(`comm-card-chg-${baseSym}`);\n"
        "        if (cardChgEl) {\n"
        "          setSafeClass(cardChgEl, `inline-flex items-center gap-1 px-2.5 py-1 rounded-xl text-xs font-mono font-bold border ${chgCls}`);\n"
        "          setSafeHtml(cardChgEl, `<span>${sign}${chgPct.toFixed(2)}%</span><span class=\"text-[10px] opacity-80\">(${sign}${chg.toFixed(2)})</span>`);\n"
        "        }\n"
        "\n"
        "        // Update Table Row LTP with visual flash\n"
        "        const rowLtpEl = document.getElementById(`comm-row-ltp-${baseSym}`);\n"
        "        if (rowLtpEl) {\n"
        "          const oldLtp = parseFloat(rowLtpEl.getAttribute('data-prev-ltp') || '0');\n"
        "          if (oldLtp === 0 || Math.abs(newLtp - oldLtp) >= 0.01) {\n"
        "            setSafeText(rowLtpEl, `₹${newLtp.toFixed(2)}`);\n"
        "            rowLtpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "            if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.01) {\n"
        "              const flashCls = newLtp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "              rowLtpEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "              setTimeout(() => rowLtpEl.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "            }\n"
        "          }\n"
        "        }\n"
        "\n"
        "        // Update Table Row Change Cell\n"
        "        const rowChgEl = document.getElementById(`comm-row-chg-${baseSym}`);\n"
        "        if (rowChgEl) {\n"
        "          setSafeClass(rowChgEl, `py-2.5 px-3 text-right ${isUp ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}`);\n"
        "          setSafeText(rowChgEl, `${sign}${chgPct.toFixed(2)}%`);\n"
        "        }"
    ).replace("\n", sep)

    if old_comm_card_update in text:
        text = text.replace(old_comm_card_update, new_comm_card_update, 1)
        print("[OK] Guarded Commodities cards and table rows against layout thrashing")
    else:
        print("ERROR: old_comm_card_update not found!")
        return

    # Update volume and OI
    old_vols = (
        "        // Update Volume and Range Meter\n"
        "        const volEl = document.getElementById(`comm-card-vol-${baseSym}`);\n"
        "        if (volEl) volEl.innerText = Number(q.volume || 0).toLocaleString();\n"
        "\n"
        "        const rowVolEl = document.getElementById(`comm-row-vol-${baseSym}`);\n"
        "        if (rowVolEl) rowVolEl.innerText = Number(q.volume || 0).toLocaleString();\n"
        "\n"
        "        const rowOiEl = document.getElementById(`comm-row-oi-${baseSym}`);\n"
        "        if (rowOiEl) rowOiEl.innerText = Number(q.open_interest || 0).toLocaleString();"
    ).replace("\n", sep)

    new_vols = (
        "        // Update Volume and Range Meter\n"
        "        setSafeText(`comm-card-vol-${baseSym}`, Number(q.volume || 0).toLocaleString());\n"
        "        setSafeText(`comm-row-vol-${baseSym}`, Number(q.volume || 0).toLocaleString());\n"
        "        setSafeText(`comm-row-oi-${baseSym}`, Number(q.open_interest || 0).toLocaleString());"
    ).replace("\n", sep)

    if old_vols in text:
        text = text.replace(old_vols, new_vols, 1)
        print("[OK] Guarded Commodities volume and OI text")

    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    print("[OK] Successfully updated index.html for commodities desk")

if __name__ == "__main__":
    main()
