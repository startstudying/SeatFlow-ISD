"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useState, type ReactNode } from "react";

const navItems = [
  { href: "/", label: "市场总览", icon: "◇" },
  { href: "/capital-graph", label: "资金传播图", icon: "⌘" },
  { href: "/predictions", label: "股票预测分析", icon: "↗" },
  { href: "/experiments", label: "实验中心", icon: "◫" }
];

const supportItems = [
  {
    id: "data-center",
    href: "/data-center",
    label: "数据中心",
    icon: "▦",
    tabs: [
      ["overview", "数据总览"],
      ["assets", "数据资产"],
      ["quality", "数据质量"],
      ["graph", "动态图谱"],
      ["versions", "数据集版本"],
      ["sync", "同步任务"]
    ]
  },
  {
    id: "settings",
    href: "/settings",
    label: "系统设置",
    icon: "⚙",
    tabs: [
      ["general", "通用设置"],
      ["data-sources", "数据源配置"],
      ["model", "模型配置"],
      ["experiment", "实验配置"],
      ["monitoring", "服务监控"],
      ["users", "用户权限"],
      ["audit", "日志审计"]
    ]
  }
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </div>
          <div>
            <h1>SeatFlow</h1>
            <p>动态时序席位异构图研究平台</p>
          </div>
        </div>

        <nav className="nav" aria-label="主导航">
          {navItems.map((item) => {
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link className={active ? "active" : ""} href={item.href} key={item.href}>
                <span className="nav-icon" aria-hidden="true">
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <nav className="nav nav-secondary" aria-label="业务模块导航">
          {supportItems.map((item) => {
            const active = pathname === item.href;
            const open = active || expanded === item.id;
            return (
              <div className="nav-group" key={item.id}>
                <button
                  aria-expanded={open}
                  className={`nav-trigger ${active ? "active" : ""}`}
                  onClick={() => setExpanded(open && !active ? null : item.id)}
                  type="button"
                >
                  <span className="nav-icon" aria-hidden="true">
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                  <b aria-hidden="true">{open ? "⌃" : "⌄"}</b>
                </button>
                {open && (
                  <div className="nav-submenu">
                    {item.tabs.map(([tab, label]) => {
                      const selected = active && searchParams.get("tab") === tab;
                      return (
                        <Link className={selected ? "selected" : ""} href={`${item.href}?tab=${tab}`} key={tab}>
                          {label}
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        <div className="status-dock">
          <span className="status-kicker">系统状态</span>
          <strong>
            <i className="status-dot status-warn" />
            真实数据与模型待验证
          </strong>
          <p>当前仅展示经接口返回的快照与证据状态，不生成交易结论。</p>
          <div className="status-meta">
            <span>模型状态：待验证</span>
            <span>页面模式：证据优先</span>
          </div>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
