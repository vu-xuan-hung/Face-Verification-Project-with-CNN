import { EmptyState } from './shared-ui';
import { BarChart3 } from 'lucide-react';

export default function ActivityChart({ activity = [] }) {
  if (!activity.length) return <EmptyState icon={BarChart3} title="No activity yet" message="Seven-day activity appears after authentication events are recorded." />;
  const max = Math.max(1, ...activity.map(day => Number(day.total) || 0));
  return <div className="activity-chart" aria-label="Granted and denied access activity over the last seven days">
    <div className="activity-chart__legend"><span><i className="legend-dot legend-dot--granted" />Granted</span><span><i className="legend-dot legend-dot--denied" />Denied</span></div>
    <div className="activity-chart__plot">
      {activity.map(day => {
        const granted = Number(day.granted) || 0;
        const denied = Number(day.denied) || 0;
        return <div className="activity-chart__column" key={day.log_date} title={`${day.log_date}: ${granted} granted, ${denied} denied`}>
          <div className="activity-chart__count">{granted + denied}</div>
          <div className="activity-chart__track">
            <span className="activity-chart__bar activity-chart__bar--denied" style={{ height: `${Math.max(denied ? 5 : 0, denied / max * 100)}%` }} />
            <span className="activity-chart__bar activity-chart__bar--granted" style={{ height: `${Math.max(granted ? 5 : 0, granted / max * 100)}%` }} />
          </div>
          <span>{day.log_date?.slice(5) || '—'}</span>
        </div>;
      })}
    </div>
  </div>;
}
