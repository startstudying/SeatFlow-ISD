export function DataTable({
  title,
  columns,
  rows,
  note
}: {
  title: string;
  columns: string[];
  rows: Array<Array<string | number>>;
  note?: string;
}) {
  return (
    <section className="card table-card">
      <div className="card-heading">
        <h3>{title}</h3>
        <span className="table-more">查看详情 ›</span>
      </div>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={`${rowIndex}-${cellIndex}`}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {note && <p className="fine">{note}</p>}
    </section>
  );
}
