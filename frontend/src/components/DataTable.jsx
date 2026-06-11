import { useState, useMemo } from 'react';

export default function DataTable({ data, columns, onDownload, isLoading }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  const cols = useMemo(() => {
    return columns?.length ? columns : Object.keys(data[0] || {});
  }, [columns, data]);

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const filteredAndSortedData = useMemo(() => {
    let result = [...data];

    // 1. Filter by search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(row => {
        return cols.some(col => {
          const val = String(row[col] ?? '').toLowerCase();
          return val.includes(q);
        });
      });
    }

    // 2. Sort by sort config
    if (sortConfig.key) {
      result.sort((a, b) => {
        const aVal = String(a[sortConfig.key] ?? '');
        const bVal = String(b[sortConfig.key] ?? '');
        
        // Try numeric sorting if possible
        const aNum = parseFloat(aVal.replace(/[^\d.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
          return sortConfig.direction === 'asc' ? aNum - bNum : bNum - aNum;
        }

        return sortConfig.direction === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      });
    }

    return result;
  }, [data, cols, searchQuery, sortConfig]);

  if (!data || data.length === 0) return null;

  const prettify = (key) =>
    key.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim()
      .replace(/\b\w/g, c => c.toUpperCase());

  // Render cell contents with custom styling for specific types
  const renderCellContent = (col, value) => {
    const strVal = String(value ?? '').trim();
    if (!strVal || strVal === '—' || strVal === 'None' || strVal === 'null') {
      return <span className="cell-empty">—</span>;
    }

    const colLower = col.toLowerCase();
    
    // Highlight Model/SKU codes
    if (colLower === 'model' || colLower === 'sku' || colLower === 'part #') {
      return <span className="cell-badge-model">{strVal}</span>;
    }
    
    // Highlight category
    if (colLower === 'category') {
      return <span className="cell-badge-category">{strVal}</span>;
    }

    // Handle array list values (like features)
    if (Array.isArray(value)) {
      return (
        <ul className="cell-list">
          {value.map((item, idx) => <li key={idx}>{item}</li>)}
        </ul>
      );
    }

    return strVal;
  };

  return (
    <div className="data-table-card">
      <div className="data-table-header">
        <div className="data-table-title">
          Extracted Intelligence
          <span className="data-table-count">
            {filteredAndSortedData.length} item{filteredAndSortedData.length !== 1 ? 's' : ''} matched 
            {searchQuery && <span className="filter-active-text"> (filtered)</span>}
          </span>
        </div>

        {/* Controls Panel */}
        <div className="data-table-controls">
          <div className="search-box-wrapper">
            <input 
              type="text" 
              className="table-search-input" 
              placeholder="Search table rows..." 
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button className="search-clear-btn" onClick={() => setSearchQuery('')}>×</button>
            )}
          </div>

          <button
            className="btn btn-success download-btn"
            onClick={onDownload}
            disabled={isLoading}
            id="download-excel-btn"
          >
            {isLoading ? (
              <><span className="spinner" style={{ width: 14, height: 14 }} /> Exporting…</>
            ) : (
              <>Export to Excel</>
            )}
          </button>
        </div>
      </div>
      
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th className="row-number-header">#</th>
              {cols.map(col => {
                const isSorted = sortConfig.key === col;
                return (
                  <th 
                    key={col} 
                    onClick={() => handleSort(col)}
                    className={`sortable-header ${isSorted ? 'sorted' : ''}`}
                  >
                    <div className="header-cell-inner">
                      <span>{prettify(col)}</span>
                      <span className="sort-arrow">
                        {!isSorted ? '↕' : sortConfig.direction === 'asc' ? '▲' : '▼'}
                      </span>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedData.length === 0 ? (
              <tr>
                <td colSpan={cols.length + 1} className="no-rows-placeholder">
                  No records match your search filter
                </td>
              </tr>
            ) : (
              filteredAndSortedData.map((row, i) => (
                <tr key={i} className="table-row-hover">
                  <td className="row-number-cell">{i + 1}</td>
                  {cols.map(col => (
                    <td key={col} className={`table-data-cell col-${col}`}>
                      {renderCellContent(col, row[col])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
