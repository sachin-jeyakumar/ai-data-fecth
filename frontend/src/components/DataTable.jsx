export default function DataTable({ data, columns, onDownload, isLoading }) {
  if (!data || data.length === 0) return null;

  const cols = columns?.length ? columns : Object.keys(data[0] || {});

  const prettify = (key) =>
    key.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim()
      .replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div className="data-table-card">
      <div className="data-table-header">
        <div className="data-table-title">
          📊 Extracted Products
          <span className="data-table-count">
            {data.length} product{data.length !== 1 ? 's' : ''} · {cols.length} field{cols.length !== 1 ? 's' : ''}
          </span>
        </div>
        <button
          className="btn btn-success"
          onClick={onDownload}
          disabled={isLoading}
          id="download-excel-btn"
        >
          {isLoading ? (
            <><span className="spinner" style={{ width: 14, height: 14 }} /> Exporting…</>
          ) : (
            <>⬇️ Download Excel</>
          )}
        </button>
      </div>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              {cols.map(col => (
                <th key={col}>{prettify(col)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{i + 1}</td>
                {cols.map(col => (
                  <td key={col} title={String(row[col] ?? '')}>
                    {String(row[col] ?? '—')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
