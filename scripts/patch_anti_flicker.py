#!/usr/bin/env python3
"""
Patch script to eliminate all UI flickering, layout thrashing, and table rebuilds
across Swing Desk, Intraday Desk, Options Desk, Commodities Desk, and Navigation Bar.
Preserves CRLF line endings.
"""

import sys

def main():
    path = "web/templates/index.html"
    with open(path, "rb") as f:
        content_bytes = f.read()

    # Determine line endings
    crlf = b"\r\n" in content_bytes
    sep = "\r\n" if crlf else "\n"
    
    text = content_bytes.decode("utf-8")
    original_len = len(text)
    print(f"Original length: {original_len}, CRLF: {crlf}")

    # =========================================================================
    # 1. ADD setSafeText, setSafeHtml, setSafeClass helpers
    # =========================================================================
    helper_code = (
        "    // High-performance DOM reconciliation helpers (Zero layout thrashing or flicker)\n"
        "    window.setSafeText = function(idOrEl, text) {\n"
        "      if (!idOrEl || text === undefined || text === null) return;\n"
        "      const el = (typeof idOrEl === 'string') ? document.getElementById(idOrEl) : idOrEl;\n"
        "      if (el && el.innerText !== String(text)) {\n"
        "        el.innerText = String(text);\n"
        "      }\n"
        "    };\n"
        "    window.setSafeHtml = function(idOrEl, html) {\n"
        "      if (!idOrEl || html === undefined || html === null) return;\n"
        "      const el = (typeof idOrEl === 'string') ? document.getElementById(idOrEl) : idOrEl;\n"
        "      if (el && el.innerHTML !== String(html)) {\n"
        "        el.innerHTML = String(html);\n"
        "      }\n"
        "    };\n"
        "    window.setSafeClass = function(idOrEl, cls) {\n"
        "      if (!idOrEl || cls === undefined || cls === null) return;\n"
        "      const el = (typeof idOrEl === 'string') ? document.getElementById(idOrEl) : idOrEl;\n"
        "      if (el && el.className !== String(cls)) {\n"
        "        el.className = String(cls);\n"
        "      }\n"
        "    };\n\n"
    ).replace("\n", sep)

    target_anchor = "    // Unified Nifty Spot / ATM Strike State Tracker (Guarantees zero flicker or jitter)"
    if "window.setSafeText" not in text:
        if target_anchor in text:
            text = text.replace(target_anchor, helper_code + target_anchor, 1)
            print("[OK] Added setSafeText, setSafeHtml, setSafeClass helpers")
        else:
            print("ERROR: Target anchor for helpers not found!")
            sys.exit(1)
    else:
        print("Helpers already present.")

    # =========================================================================
    # 2. FIX POLLER COLLISIONS & REFERENCE ERROR IN setInterval
    # =========================================================================
    old_poller_code = (
        "        // 1. High-frequency 4-Second Live Spot Tick Streamer\n"
        "        setInterval(() => {\n"
        "          if (!document.hidden && isKeepaliveWindowActive && isClientInMarketKeepaliveWindow()) {\n"
        "            pollLiveSpotTick();\n"
        "          }\n"
        "        }, 4000);\n"
        "\n"
        "        // 2. 15-Second Options & Intraday Streamer\n"
        "        setInterval(() => {\n"
        "          if (!document.hidden && isKeepaliveWindowActive && isClientInMarketKeepaliveWindow()) {\n"
        "            const optMod = document.getElementById('module-options');\n"
        "            const intraMod = document.getElementById('module-intraday');\n"
        "            if (optMod && !optMod.classList.contains('hidden')) {\n"
        "              loadOptionsChainDesk(false);\n"
        "            } else if (intraMod && !intraMod.classList.contains('hidden')) {\n"
        "              loadIntradayExpiryProgression(false);\n"
        "            }\n"
        "          }\n"
        "        }, 15000);"
    ).replace("\n", sep)

    new_poller_code = (
        "        // 1. High-frequency 4-Second Live Spot Tick Streamer (Harmonized with Kotak Neo continuous stream)\n"
        "        setInterval(() => {\n"
        "          if (!document.hidden && isKeepaliveWindowActive && isClientInMarketKeepaliveWindow()) {\n"
        "            if (!window.isKotakStreamActive) {\n"
        "              pollLiveSpotTick();\n"
        "            }\n"
        "          }\n"
        "        }, 4000);\n"
        "\n"
        "        // 2. 15-Second Options & Intraday Streamer (Runs only if continuous live stream is not active)\n"
        "        setInterval(() => {\n"
        "          if (!document.hidden && isKeepaliveWindowActive && isClientInMarketKeepaliveWindow()) {\n"
        "            const optMod = document.getElementById('module-options');\n"
        "            const intraMod = document.getElementById('module-intraday');\n"
        "            if (optMod && !optMod.classList.contains('hidden')) {\n"
        "              if (!window.isKotakStreamActive) loadOptionsChainDesk(false);\n"
        "            } else if (intraMod && !intraMod.classList.contains('hidden')) {\n"
        "              if (!window.isKotakStreamActive) loadIntradayExpiryDesk(false);\n"
        "            }\n"
        "          }\n"
        "        }, 15000);"
    ).replace("\n", sep)

    if old_poller_code in text:
        text = text.replace(old_poller_code, new_poller_code, 1)
        print("[OK] Harmonized poller intervals and fixed ReferenceError")
    else:
        print("Notice: old_poller_code not matched exactly; checking partials...")
        if "loadIntradayExpiryProgression(false);" in text:
            text = text.replace("loadIntradayExpiryProgression(false);", "loadIntradayExpiryDesk(false);")
            print("[OK] Fixed loadIntradayExpiryProgression reference error")

    # =========================================================================
    # 3. PREVENT TABLE BLINKS IN renderOptionsChainUI
    # =========================================================================
    old_opt_table_render = (
        "      const tbody = document.getElementById('opt-chain-tbody');\n"
        "      if (!tbody) return;\n"
        "      if (strikes.length === 0) {\n"
        "        tbody.innerHTML = `<tr><td colspan=\"14\" class=\"p-8 text-center text-slate-500 font-sans\">No strike rows available in selected range.</td></tr>`;\n"
        "        return;\n"
        "      }\n"
        "\n"
        "      const maxOi = Math.max(1, ...strikes.map(s => Math.max(s.ce_oi || 0, s.pe_oi || 0)));\n"
        "      const formatBuildupBadge = (b, isCall) => {\n"
        "        if (b === 'SHORT_BUILDUP') {\n"
        "          // Option Writing (Resistance on Call side, Support on Put side)\n"
        "          const cls = isCall ? 'bg-rose-500/20 text-rose-300 border-rose-500/30' : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';\n"
        "          const lbl = isCall ? 'CALL WRITING' : 'PUT WRITING';\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border ${cls} font-bold whitespace-nowrap\">${lbl}</span>`;\n"
        "        }\n"
        "        if (b === 'LONG_BUILDUP') {\n"
        "          const cls = isCall ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' : 'bg-rose-500/20 text-rose-300 border-rose-500/30';\n"
        "          const lbl = isCall ? 'CALL BUYING' : 'PUT BUYING';\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border ${cls} whitespace-nowrap\">${lbl}</span>`;\n"
        "        }\n"
        "        if (b === 'SHORT_COVERING') {\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border border-cyan-500/30 bg-cyan-500/15 text-cyan-300 whitespace-nowrap\">SHORT COVER</span>`;\n"
        "        }\n"
        "        if (b === 'LONG_UNWINDING') {\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border border-amber-500/30 bg-amber-500/15 text-amber-300 whitespace-nowrap\">UNWINDING</span>`;\n"
        "        }\n"
        "        return `<span class=\"quant-badge inline-flex items-center text-[9px] text-slate-500 whitespace-nowrap\">NEUTRAL</span>`;\n"
        "      };\n"
        "\n"
        "      tbody.innerHTML = strikes.map(s => {"
    ).replace("\n", sep)

    new_opt_table_render = (
        "      const tbody = document.getElementById('opt-chain-tbody');\n"
        "      if (!tbody) return;\n"
        "      if (strikes.length === 0) {\n"
        "        setSafeHtml(tbody, `<tr><td colspan=\"14\" class=\"p-8 text-center text-slate-500 font-sans\">No strike rows available in selected range.</td></tr>`);\n"
        "        return;\n"
        "      }\n"
        "\n"
        "      const maxOi = Math.max(1, ...strikes.map(s => Math.max(s.ce_oi || 0, s.pe_oi || 0)));\n"
        "      const formatBuildupBadge = (b, isCall) => {\n"
        "        if (b === 'SHORT_BUILDUP') {\n"
        "          const cls = isCall ? 'bg-rose-500/20 text-rose-300 border-rose-500/30' : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';\n"
        "          const lbl = isCall ? 'CALL WRITING' : 'PUT WRITING';\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border ${cls} font-bold whitespace-nowrap\">${lbl}</span>`;\n"
        "        }\n"
        "        if (b === 'LONG_BUILDUP') {\n"
        "          const cls = isCall ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' : 'bg-rose-500/20 text-rose-300 border-rose-500/30';\n"
        "          const lbl = isCall ? 'CALL BUYING' : 'PUT BUYING';\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border ${cls} whitespace-nowrap\">${lbl}</span>`;\n"
        "        }\n"
        "        if (b === 'SHORT_COVERING') {\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border border-cyan-500/30 bg-cyan-500/15 text-cyan-300 whitespace-nowrap\">SHORT COVER</span>`;\n"
        "        }\n"
        "        if (b === 'LONG_UNWINDING') {\n"
        "          return `<span class=\"quant-badge inline-flex items-center px-1.5 py-0.5 rounded text-[9px] border border-amber-500/30 bg-amber-500/15 text-amber-300 whitespace-nowrap\">UNWINDING</span>`;\n"
        "        }\n"
        "        return `<span class=\"quant-badge inline-flex items-center text-[9px] text-slate-500 whitespace-nowrap\">NEUTRAL</span>`;\n"
        "      };\n"
        "\n"
        "      const chainRenderKey = `${selectedOptionsExpiryType}-${optChainStrikeRange}-${strikes.length}`;\n"
        "      const prevKey = tbody.getAttribute('data-chain-key');\n"
        "      const tableHasMatchingRows = (prevKey === chainRenderKey) && (tbody.children.length === strikes.length);\n"
        "\n"
        "      if (tableHasMatchingRows) {\n"
        "        // Fast in-place cell updates without wiping DOM tree (Eliminates table blinking)\n"
        "        strikes.forEach(s => {\n"
        "          setSafeText(`opt-ce-oi-${s.strike}`, (s.ce_oi || 0).toLocaleString('en-IN'));\n"
        "          setSafeText(`opt-ce-vol-${s.strike}`, (s.ce_vol || 0).toLocaleString('en-IN'));\n"
        "          setSafeText(`opt-pe-vol-${s.strike}`, (s.pe_vol || 0).toLocaleString('en-IN'));\n"
        "          setSafeText(`opt-pe-oi-${s.strike}`, (s.pe_oi || 0).toLocaleString('en-IN'));\n"
        "          const ceLtpEl = document.getElementById(`opt-ce-ltp-${s.strike}`);\n"
        "          if (ceLtpEl) {\n"
        "            const oldLtp = parseFloat(ceLtpEl.getAttribute('data-prev-ltp') || '0');\n"
        "            const newLtp = Number(s.ce_ltp || 0);\n"
        "            if (oldLtp === 0 || Math.abs(newLtp - oldLtp) >= 0.01) {\n"
        "              const pchg = s.ce_pchg || 0;\n"
        "              const sign = pchg >= 0 ? '+' : '';\n"
        "              const pchgCls = pchg >= 0 ? 'text-emerald-400' : 'text-rose-400';\n"
        "              ceLtpEl.innerHTML = `₹${newLtp.toFixed(2)} <span id=\"opt-ce-pchg-${s.strike}\" class=\"text-[10px] ${pchgCls}\">(${sign}${Number(pchg).toFixed(0)}%)</span>`;\n"
        "              ceLtpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "            }\n"
        "          }\n"
        "          const peLtpEl = document.getElementById(`opt-pe-ltp-${s.strike}`);\n"
        "          if (peLtpEl) {\n"
        "            const oldLtp = parseFloat(peLtpEl.getAttribute('data-prev-ltp') || '0');\n"
        "            const newLtp = Number(s.pe_ltp || 0);\n"
        "            if (oldLtp === 0 || Math.abs(newLtp - oldLtp) >= 0.01) {\n"
        "              const pchg = s.pe_pchg || 0;\n"
        "              const sign = pchg >= 0 ? '+' : '';\n"
        "              const pchgCls = pchg >= 0 ? 'text-emerald-400' : 'text-rose-400';\n"
        "              peLtpEl.innerHTML = `₹${newLtp.toFixed(2)} <span id=\"opt-pe-pchg-${s.strike}\" class=\"text-[10px] ${pchgCls}\">(${sign}${Number(pchg).toFixed(0)}%)</span>`;\n"
        "              peLtpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "            }\n"
        "          }\n"
        "        });\n"
        "      } else {\n"
        "        tbody.innerHTML = strikes.map(s => {"
    ).replace("\n", sep)

    if old_opt_table_render in text:
        text = text.replace(old_opt_table_render, new_opt_table_render, 1)
        print("[OK] Guarded Option Chain table against background wipes")
        
        # Close the else branch after .join('');
        old_join = (
            "      }).join('');\n"
            "\n"
            "      // Synchronize strategy builder state without overwriting reset/cleared builder"
        ).replace("\n", sep)
        new_join = (
            "        }).join('');\n"
            "        tbody.setAttribute('data-chain-key', chainRenderKey);\n"
            "        if (optContainerEl && optPrevScrollTop > 0) {\n"
            "          optContainerEl.scrollTop = optPrevScrollTop;\n"
            "        }\n"
            "      }\n"
            "\n"
            "      // Synchronize strategy builder state without overwriting reset/cleared builder"
        ).replace("\n", sep)
        if old_join in text:
            text = text.replace(old_join, new_join, 1)
            print("[OK] Closed else block for option chain render and added scroll restoration")
        else:
            print("ERROR: old_join anchor not found!")
    else:
        print("Notice: old_opt_table_render not found; checking if already patched...")

    # Gate drawPayoffGraph in renderOptionsChainUI
    old_payoff_call = (
        "      // Synchronize strategy builder state without overwriting reset/cleared builder\n"
        "      if (typeof renderBuilderCompactChain === 'function') {\n"
        "        renderBuilderCompactChain();\n"
        "      }\n"
        "      recalculateStrategyMetrics();\n"
        "      drawPayoffGraph();"
    ).replace("\n", sep)
    new_payoff_call = (
        "      // Synchronize strategy builder state only if builder has legs\n"
        "      if (typeof strategyLegs !== 'undefined' && strategyLegs && strategyLegs.length > 0) {\n"
        "        if (typeof renderBuilderCompactChain === 'function') {\n"
        "          renderBuilderCompactChain();\n"
        "        }\n"
        "        recalculateStrategyMetrics();\n"
        "        drawPayoffGraph();\n"
        "      }"
    ).replace("\n", sep)
    if old_payoff_call in text:
        text = text.replace(old_payoff_call, new_payoff_call, 1)
        print("[OK] Gated payoff graph canvas redraw to avoid unnecessary redraw flickering")

    # =========================================================================
    # 4. UPGRADE pollContinuousKotakStream WITH EQUALITY GUARDS
    # =========================================================================
    old_kotak_spot_code = (
        "            if (intra.nifty_spot > 0) {\n"
        "              const spot = Number(intra.nifty_spot);\n"
        "              const formatted = '₹' + spot.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });\n"
        "              updateNiftySpotAtmCard(spot, null, null, null);\n"
        "              const intraBadge = document.getElementById('intraday-spot-badge');\n"
        "              if (intraBadge) intraBadge.innerText = `NIFTY: ${formatted}`;\n"
        "\n"
        "              updateOptionsDeskStreamIndicator(spot, data.is_market_open, 'KOTAK_NEO_STREAM', tickCount);\n"
        "\n"
        "              // Live Intraday Desk Top KPIs & Analysis Synchronization via unified non-flickering controller\n"
        "              const isMarketOpen = Boolean(data.is_market_open);\n"
        "              const isLiveStreaming = Boolean(data.is_streaming && isMarketOpen);\n"
        "              const curAtm = intra.atm_strike || Math.round(spot / 50) * 50;\n"
        "              updateIntradaySpotAtmCard(spot, curAtm, isLiveStreaming, null, null);"
    ).replace("\n", sep)

    new_kotak_spot_code = (
        "            if (intra.nifty_spot > 0) {\n"
        "              const spot = Number(intra.nifty_spot);\n"
        "              const formatted = '₹' + spot.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });\n"
        "              const prevClose = Number(intra.prev_close || data.prev_close || 0);\n"
        "              const chgPts = Number(intra.change_pts !== undefined ? intra.change_pts : (data.change_pts || 0));\n"
        "              const chgPct = Number(intra.change_pct !== undefined ? intra.change_pct : (data.change_pct || 0));\n"
        "              let diffStr = '';\n"
        "              const diffVal = (chgPts !== 0 || chgPct !== 0) ? chgPts : (prevClose > 0 ? (spot - prevClose) : 0);\n"
        "              const diffPctVal = (chgPts !== 0 || chgPct !== 0) ? chgPct : (prevClose > 0 ? ((diffVal / prevClose) * 100) : 0);\n"
        "              if (diffVal !== 0 || diffPctVal !== 0) {\n"
        "                const sign = diffVal >= 0 ? '+' : '';\n"
        "                const cls = diffVal >= 0 ? 'text-emerald-400' : 'text-rose-400';\n"
        "                diffStr = `<span class=\"${cls} font-bold\">${sign}${diffVal.toFixed(1)} (${sign}${diffPctVal.toFixed(2)}%)</span>`;\n"
        "              }\n"
        "              updateNiftySpotAtmCard(spot, intra.atm_strike, (niftyOptionsChainCache?.weekly?.expiry), diffStr, prevClose);\n"
        "              const intraBadge = document.getElementById('intraday-spot-badge');\n"
        "              if (intraBadge) {\n"
        "                const newBadgeHtml = `NIFTY: ${formatted} ${diffStr}`;\n"
        "                setSafeHtml(intraBadge, newBadgeHtml);\n"
        "              }\n"
        "\n"
        "              updateOptionsDeskStreamIndicator(spot, data.is_market_open, 'KOTAK_NEO_STREAM', tickCount);\n"
        "\n"
        "              // Live Intraday Desk Top KPIs & Analysis Synchronization via unified non-flickering controller\n"
        "              const isMarketOpen = Boolean(data.is_market_open);\n"
        "              const isLiveStreaming = Boolean(data.is_streaming && isMarketOpen);\n"
        "              window.isKotakStreamActive = isLiveStreaming;\n"
        "              const curAtm = intra.atm_strike || Math.round(spot / 50) * 50;\n"
        "              updateIntradaySpotAtmCard(spot, curAtm, isLiveStreaming, diffStr, null);"
    ).replace("\n", sep)

    if old_kotak_spot_code in text:
        text = text.replace(old_kotak_spot_code, new_kotak_spot_code, 1)
        print("[OK] Synchronized spot diff string in pollContinuousKotakStream")
    else:
        print("Notice: old_kotak_spot_code not found")

    # Guard Option Chain strike ticks in pollContinuousKotakStream
    old_opt_ticks_poll = (
        "                const ltpEl = document.getElementById(`opt-${type}-ltp-${strike}`);\n"
        "                if (ltpEl && opt.ltp) {\n"
        "                  const oldLtp = parseFloat(ltpEl.getAttribute('data-prev-ltp') || '0');\n"
        "                  const newLtp = Number(opt.ltp);\n"
        "                  const pchg = opt.change_pct !== undefined ? opt.change_pct : 0;\n"
        "                  const sign = pchg >= 0 ? '+' : '';\n"
        "                  const pchgCls = pchg >= 0 ? 'text-emerald-400' : 'text-rose-400';\n"
        "                  ltpEl.innerHTML = `₹${newLtp.toFixed(2)} <span id=\"opt-${type}-pchg-${strike}\" class=\"text-[10px] ${pchgCls}\">(${sign}${Number(pchg).toFixed(0)}%)</span>`;\n"
        "                  ltpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "                  if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.05) {\n"
        "                    const flashCls = newLtp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "                    ltpEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "                    setTimeout(() => ltpEl.classList.remove(...flashCls, 'rounded', 'px-1'), 800);\n"
        "                  }\n"
        "                }\n"
        "\n"
        "                const oiEl = document.getElementById(`opt-${type}-oi-${strike}`);\n"
        "                if (oiEl && opt.oi !== undefined && opt.oi !== null) {\n"
        "                  oiEl.innerText = Number(opt.oi).toLocaleString('en-IN');\n"
        "                }\n"
        "\n"
        "                const volEl = document.getElementById(`opt-${type}-vol-${strike}`);\n"
        "                if (volEl && opt.volume !== undefined && opt.volume !== null) {\n"
        "                  volEl.innerText = Number(opt.volume).toLocaleString('en-IN');\n"
        "                }"
    ).replace("\n", sep)

    new_opt_ticks_poll = (
        "                const ltpEl = document.getElementById(`opt-${type}-ltp-${strike}`);\n"
        "                if (ltpEl && opt.ltp) {\n"
        "                  const oldLtp = parseFloat(ltpEl.getAttribute('data-prev-ltp') || '0');\n"
        "                  const newLtp = Number(opt.ltp);\n"
        "                  if (oldLtp === 0 || Math.abs(newLtp - oldLtp) >= 0.01) {\n"
        "                    const pchg = opt.change_pct !== undefined ? opt.change_pct : 0;\n"
        "                    const sign = pchg >= 0 ? '+' : '';\n"
        "                    const pchgCls = pchg >= 0 ? 'text-emerald-400' : 'text-rose-400';\n"
        "                    ltpEl.innerHTML = `₹${newLtp.toFixed(2)} <span id=\"opt-${type}-pchg-${strike}\" class=\"text-[10px] ${pchgCls}\">(${sign}${Number(pchg).toFixed(0)}%)</span>`;\n"
        "                    ltpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "                    if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.05) {\n"
        "                      const flashCls = newLtp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "                      ltpEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "                      setTimeout(() => ltpEl.classList.remove(...flashCls, 'rounded', 'px-1'), 800);\n"
        "                    }\n"
        "                  }\n"
        "                }\n"
        "\n"
        "                if (opt.oi !== undefined && opt.oi !== null) {\n"
        "                  setSafeText(`opt-${type}-oi-${strike}`, Number(opt.oi).toLocaleString('en-IN'));\n"
        "                }\n"
        "\n"
        "                if (opt.volume !== undefined && opt.volume !== null) {\n"
        "                  setSafeText(`opt-${type}-vol-${strike}`, Number(opt.volume).toLocaleString('en-IN'));\n"
        "                }"
    ).replace("\n", sep)

    if old_opt_ticks_poll in text:
        text = text.replace(old_opt_ticks_poll, new_opt_ticks_poll, 1)
        print("[OK] Guarded Option Chain strike ticks in pollContinuousKotakStream")
    else:
        print("Notice: old_opt_ticks_poll not found")

    # Guard Swing Position Row updates in pollContinuousKotakStream
    old_swing_pos_poll = (
        "                if (ltpEl && pos.ltp) {\n"
        "                  const oldLtp = parseFloat(ltpEl.getAttribute('data-prev-ltp') || '0');\n"
        "                  const newLtp = Number(pos.ltp);\n"
        "                  ltpEl.innerText = `LTP: ₹${newLtp.toFixed(2)}`;\n"
        "                  ltpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "                  if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.01) {\n"
        "                    const flashCls = newLtp >= oldLtp ? 'text-emerald-400 font-extrabold' : 'text-rose-400 font-extrabold';\n"
        "                    ltpEl.className = `text-[11px] ${flashCls}`;\n"
        "                    if (pnlCell) {\n"
        "                      const pnlBg = newLtp >= oldLtp ? 'bg-emerald-500/20' : 'bg-rose-500/20';\n"
        "                      pnlCell.classList.add(pnlBg);\n"
        "                      setTimeout(() => pnlCell.classList.remove(pnlBg), 900);\n"
        "                    }\n"
        "                    setTimeout(() => {\n"
        "                      ltpEl.className = 'text-[11px] font-bold text-white';\n"
        "                    }, 900);\n"
        "                  }\n"
        "                }\n"
        "\n"
        "                if (valEl && pos.position_value) {\n"
        "                  valEl.innerText = `₹${Math.round(pos.position_value).toLocaleString('en-IN')}`;\n"
        "                }\n"
        "\n"
        "                const pnl = pos.unrealized_pnl || 0;\n"
        "                const pnlPct = pos.pnl_pct || 0;\n"
        "                const isPos = pnl >= 0;\n"
        "                const sign = isPos ? '+' : '';\n"
        "\n"
        "                if (pnlCell) {\n"
        "                  pnlCell.className = `p-3.5 ${isPos ? 'text-emerald-400' : 'text-rose-400'} font-bold font-mono transition-colors duration-300`;\n"
        "                }\n"
        "                if (pnlEl) {\n"
        "                  pnlEl.innerText = `${sign}₹${pnl.toFixed(2)}`;\n"
        "                }\n"
        "                if (pnlPctEl) {\n"
        "                  pnlPctEl.innerText = `(${sign}${pnlPct.toFixed(2)}%)`;\n"
        "                }\n"
        "                if (rEl && pos.current_r !== undefined) {\n"
        "                  const rVal = Number(pos.current_r || 0);\n"
        "                  rEl.className = `p-3.5 font-bold ${rVal >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;\n"
        "                  rEl.innerText = `${rVal >= 0 ? '+' : ''}${rVal.toFixed(2)} R`;\n"
        "                }\n"
        "                if (timeEl && pos.updated_at) {\n"
        "                  const tStr = String(pos.updated_at).slice(11, 19);\n"
        "                  timeEl.innerText = `LTP @ ${tStr || 'Live'} IST`;\n"
        "                }\n"
        "                if (dayChgEl && pos.day_change_pct !== null && pos.day_change_pct !== undefined) {\n"
        "                  const dc = Number(pos.day_change_pct);\n"
        "                  const dcSign = dc >= 0 ? '+' : '';\n"
        "                  const dcArr = dc >= 0 ? '▲' : '▼';\n"
        "                  const dcCls = dc >= 0 ? 'text-emerald-400 bg-emerald-950/70 border-emerald-500/30' : 'text-rose-400 bg-rose-950/70 border-rose-500/30';\n"
        "                  dayChgEl.className = `text-[10px] font-bold font-mono px-1.5 py-0.2 rounded border ${dcCls}`;\n"
        "                  dayChgEl.innerText = `${dcArr} ${dcSign}${dc.toFixed(2)}%`;\n"
        "                }"
    ).replace("\n", sep)

    new_swing_pos_poll = (
        "                if (ltpEl && pos.ltp) {\n"
        "                  const oldLtp = parseFloat(ltpEl.getAttribute('data-prev-ltp') || '0');\n"
        "                  const newLtp = Number(pos.ltp);\n"
        "                  if (oldLtp === 0 || Math.abs(newLtp - oldLtp) >= 0.01) {\n"
        "                    setSafeText(ltpEl, `LTP: ₹${newLtp.toFixed(2)}`);\n"
        "                    ltpEl.setAttribute('data-prev-ltp', newLtp);\n"
        "\n"
        "                    if (oldLtp > 0 && Math.abs(newLtp - oldLtp) > 0.01) {\n"
        "                      const flashCls = newLtp >= oldLtp ? 'text-emerald-400 font-extrabold' : 'text-rose-400 font-extrabold';\n"
        "                      ltpEl.className = `text-[11px] ${flashCls}`;\n"
        "                      if (pnlCell) {\n"
        "                        const pnlBg = newLtp >= oldLtp ? 'bg-emerald-500/20' : 'bg-rose-500/20';\n"
        "                        pnlCell.classList.add(pnlBg);\n"
        "                        setTimeout(() => pnlCell.classList.remove(pnlBg), 900);\n"
        "                      }\n"
        "                      setTimeout(() => {\n"
        "                        ltpEl.className = 'text-[11px] font-bold text-white';\n"
        "                      }, 900);\n"
        "                    }\n"
        "                  }\n"
        "                }\n"
        "\n"
        "                if (valEl && pos.position_value) {\n"
        "                  setSafeText(valEl, `₹${Math.round(pos.position_value).toLocaleString('en-IN')}`);\n"
        "                }\n"
        "\n"
        "                const pnl = pos.unrealized_pnl || 0;\n"
        "                const pnlPct = pos.pnl_pct || 0;\n"
        "                const isPos = pnl >= 0;\n"
        "                const sign = isPos ? '+' : '';\n"
        "\n"
        "                if (pnlCell) {\n"
        "                  setSafeClass(pnlCell, `p-3.5 ${isPos ? 'text-emerald-400' : 'text-rose-400'} font-bold font-mono transition-colors duration-300`);\n"
        "                }\n"
        "                if (pnlEl) {\n"
        "                  setSafeText(pnlEl, `${sign}₹${pnl.toFixed(2)}`);\n"
        "                }\n"
        "                if (pnlPctEl) {\n"
        "                  setSafeText(pnlPctEl, `(${sign}${pnlPct.toFixed(2)}%)`);\n"
        "                }\n"
        "                if (rEl && pos.current_r !== undefined) {\n"
        "                  const rVal = Number(pos.current_r || 0);\n"
        "                  setSafeClass(rEl, `p-3.5 font-bold ${rVal >= 0 ? 'text-emerald-400' : 'text-rose-400'}`);\n"
        "                  setSafeText(rEl, `${rVal >= 0 ? '+' : ''}${rVal.toFixed(2)} R`);\n"
        "                }\n"
        "                if (timeEl && pos.updated_at) {\n"
        "                  const tStr = String(pos.updated_at).slice(11, 19);\n"
        "                  setSafeText(timeEl, `LTP @ ${tStr || 'Live'} IST`);\n"
        "                }\n"
        "                if (dayChgEl && pos.day_change_pct !== null && pos.day_change_pct !== undefined) {\n"
        "                  const dc = Number(pos.day_change_pct);\n"
        "                  const dcSign = dc >= 0 ? '+' : '';\n"
        "                  const dcArr = dc >= 0 ? '▲' : '▼';\n"
        "                  const dcCls = dc >= 0 ? 'text-emerald-400 bg-emerald-950/70 border-emerald-500/30' : 'text-rose-400 bg-rose-950/70 border-rose-500/30';\n"
        "                  setSafeClass(dayChgEl, `text-[10px] font-bold font-mono px-1.5 py-0.2 rounded border ${dcCls}`);\n"
        "                  setSafeText(dayChgEl, `${dcArr} ${dcSign}${dc.toFixed(2)}%`);\n"
        "                }"
    ).replace("\n", sep)

    if old_swing_pos_poll in text:
        text = text.replace(old_swing_pos_poll, new_swing_pos_poll, 1)
        print("[OK] Guarded Swing position rows in pollContinuousKotakStream")
    else:
        print("Notice: old_swing_pos_poll not found")

    # Guard Equity ticks in pollContinuousKotakStream
    old_eq_ticks = (
        "              const sigEl = document.getElementById(`sig-ltp-${sym}`);\n"
        "              if (sigEl) {\n"
        "                const oldLtp = parseFloat(sigEl.getAttribute('data-prev-ltp') || '0');\n"
        "                sigEl.innerText = `₹${ltp.toFixed(2)}`;\n"
        "                sigEl.setAttribute('data-prev-ltp', ltp);\n"
        "                if (oldLtp > 0 && Math.abs(ltp - oldLtp) > 0.01) {\n"
        "                  const flashCls = ltp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300', 'font-extrabold'] : ['bg-rose-500/30', 'text-rose-300', 'font-extrabold'];\n"
        "                  sigEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "                  setTimeout(() => sigEl.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "                }\n"
        "              }\n"
        "\n"
        "              const p3El = document.getElementById(`p3-ltp-${sym}`);\n"
        "              if (p3El) {\n"
        "                const oldLtp = parseFloat(p3El.getAttribute('data-prev-ltp') || '0');\n"
        "                p3El.innerText = `₹${ltp.toFixed(2)}`;\n"
        "                p3El.setAttribute('data-prev-ltp', ltp);\n"
        "                if (oldLtp > 0 && Math.abs(ltp - oldLtp) > 0.01) {\n"
        "                  const flashCls = ltp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "                  p3El.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "                  setTimeout(() => p3El.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "                }\n"
        "              }"
    ).replace("\n", sep)

    new_eq_ticks = (
        "              const sigEl = document.getElementById(`sig-ltp-${sym}`);\n"
        "              if (sigEl) {\n"
        "                const oldLtp = parseFloat(sigEl.getAttribute('data-prev-ltp') || '0');\n"
        "                if (oldLtp === 0 || Math.abs(ltp - oldLtp) >= 0.01) {\n"
        "                  setSafeText(sigEl, `₹${ltp.toFixed(2)}`);\n"
        "                  sigEl.setAttribute('data-prev-ltp', ltp);\n"
        "                  if (oldLtp > 0 && Math.abs(ltp - oldLtp) > 0.01) {\n"
        "                    const flashCls = ltp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300', 'font-extrabold'] : ['bg-rose-500/30', 'text-rose-300', 'font-extrabold'];\n"
        "                    sigEl.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "                    setTimeout(() => sigEl.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "                  }\n"
        "                }\n"
        "              }\n"
        "\n"
        "              const p3El = document.getElementById(`p3-ltp-${sym}`);\n"
        "              if (p3El) {\n"
        "                const oldLtp = parseFloat(p3El.getAttribute('data-prev-ltp') || '0');\n"
        "                if (oldLtp === 0 || Math.abs(ltp - oldLtp) >= 0.01) {\n"
        "                  setSafeText(p3El, `₹${ltp.toFixed(2)}`);\n"
        "                  p3El.setAttribute('data-prev-ltp', ltp);\n"
        "                  if (oldLtp > 0 && Math.abs(ltp - oldLtp) > 0.01) {\n"
        "                    const flashCls = ltp >= oldLtp ? ['bg-emerald-500/30', 'text-emerald-300'] : ['bg-rose-500/30', 'text-rose-300'];\n"
        "                    p3El.classList.add(...flashCls, 'rounded', 'px-1');\n"
        "                    setTimeout(() => p3El.classList.remove(...flashCls, 'rounded', 'px-1'), 750);\n"
        "                  }\n"
        "                }\n"
        "              }"
    ).replace("\n", sep)

    if old_eq_ticks in text:
        text = text.replace(old_eq_ticks, new_eq_ticks, 1)
        print("[OK] Guarded Equity ticks in pollContinuousKotakStream")
    else:
        print("Notice: old_eq_ticks not found")

    # Guard Level-2 ladder table in renderIntradayExpiryUI
    old_ladder_code = (
        "        if (dLadderTbody && Array.isArray(depthData.near_atm_strikes) && depthData.near_atm_strikes.length > 0) {\n"
        "          dLadderTbody.innerHTML = depthData.near_atm_strikes.map(row => {"
    ).replace("\n", sep)
    new_ladder_code = (
        "        if (dLadderTbody && Array.isArray(depthData.near_atm_strikes) && depthData.near_atm_strikes.length > 0) {\n"
        "          const depthSig = depthData.near_atm_strikes.map(r => `${r.strike}-${r.ce?.ltp}-${r.pe?.ltp}-${r.ce?.total_buy}-${r.pe?.total_buy}`).join('|');\n"
        "          if (dLadderTbody.getAttribute('data-depth-sig') !== depthSig) {\n"
        "            dLadderTbody.setAttribute('data-depth-sig', depthSig);\n"
        "            dLadderTbody.innerHTML = depthData.near_atm_strikes.map(row => {"
    ).replace("\n", sep)
    if old_ladder_code in text:
        text = text.replace(old_ladder_code, new_ladder_code, 1)
        # Close the if branch after }).join('');
        old_ladder_join = (
            "            `;\n"
            "          }).join('');\n"
            "        }\n"
            "      } else {"
        ).replace("\n", sep)
        new_ladder_join = (
            "            `;\n"
            "          }).join('');\n"
            "          }\n"
            "        }\n"
            "      } else {"
        ).replace("\n", sep)
        if old_ladder_join in text:
            text = text.replace(old_ladder_join, new_ladder_join, 1)
            print("[OK] Guarded Level-2 ladder table against unnecessary repaints")
        else:
            print("Notice: old_ladder_join not found")

    # Guard Snapshots Table in renderIntradayExpiryUI
    old_snaps_tbody = (
        "      // Snapshots Table\n"
        "      const tbody = document.getElementById('intra-snapshots-tbody');\n"
        "      const snaps = data.snapshots || [];\n"
        "      if (tbody) {\n"
        "        if (snaps.length === 0) {\n"
        "          tbody.innerHTML = `<tr><td colspan=\"14\" class=\"p-8 text-center text-slate-500 font-sans\">No ${currentIntradayInterval} snapshots recorded yet today. Click \"Capture Slot Now\".</td></tr>`;\n"
        "        } else {\n"
        "          tbody.innerHTML = snaps.map(s => {"
    ).replace("\n", sep)
    new_snaps_tbody = (
        "      // Snapshots Table\n"
        "      const tbody = document.getElementById('intra-snapshots-tbody');\n"
        "      const snaps = data.snapshots || [];\n"
        "      if (tbody) {\n"
        "        if (snaps.length === 0) {\n"
        "          setSafeHtml(tbody, `<tr><td colspan=\"14\" class=\"p-8 text-center text-slate-500 font-sans\">No ${currentIntradayInterval} snapshots recorded yet today. Click \"Capture Slot Now\".</td></tr>`);\n"
        "        } else {\n"
        "          const snapsSig = snaps.map(s => `${s.slot_time}-${s.spot_price}-${s.straddle_premium}-${s.near_atm_pcr}`).join('|');\n"
        "          if (tbody.getAttribute('data-snaps-sig') !== snapsSig) {\n"
        "            tbody.setAttribute('data-snaps-sig', snapsSig);\n"
        "            tbody.innerHTML = snaps.map(s => {"
    ).replace("\n", sep)
    if old_snaps_tbody in text:
        text = text.replace(old_snaps_tbody, new_snaps_tbody, 1)
        old_snaps_join = (
            "              </tr>\n"
            "            `;\n"
            "          }).join('');\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "\n"
            "    // =========================================================================\n"
            "    // MODULE 3: NIFTY 50 T+1 & T+5 MULTI-PILLAR QUANT PREDICTIONS DESK"
        ).replace("\n", sep)
        new_snaps_join = (
            "              </tr>\n"
            "            `;\n"
            "          }).join('');\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "\n"
            "    // =========================================================================\n"
            "    // MODULE 3: NIFTY 50 T+1 & T+5 MULTI-PILLAR QUANT PREDICTIONS DESK"
        ).replace("\n", sep)
        if old_snaps_join in text:
            text = text.replace(old_snaps_join, new_snaps_join, 1)
            print("[OK] Guarded Intraday snapshots table against unnecessary DOM rewrites")
        else:
            print("Notice: old_snaps_join not found")

    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))

    print(f"File updated successfully! New length: {len(text)} (delta: {len(text) - original_len})")

if __name__ == "__main__":
    main()
