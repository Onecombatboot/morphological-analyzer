/*
 * Optional API location override.
 *
 * Leave this file as-is for the normal case. The application serves the interface and the API
 * from the same origin, so the interface finds the API automatically on whatever port the
 * server was started on. Nothing here needs changing when the port changes.
 *
 * Set this ONLY if the interface is served from somewhere other than the application itself
 * (for example opened directly from disk, or hosted on a separate web server):
 *
 *   window.GMA_API_BASE = 'http://127.0.0.1:8090/api';
 *
 * A one-off redirect without editing the file: append ?api=http://127.0.0.1:8090/api to the URL.
 */
