/**
 * Metis Pyle - WebRTC Companion Client
 * Manages RTCPeerConnection and multiplexed DataChannels:
 *   - 'control': JSON-RPC 2.0 (heartbeat, auth, Phylax biometric approvals)
 *   - 'stream': Real-time SSE event stream (Lochos squad state, chat tokens)
 *   - 'terminal': Bidirectional ANSI terminal stream
 */

class PyleWebRTCClient {
    constructor(options = {}) {
        this.apiBase = options.apiBase || '';
        this.iceServers = options.iceServers || [{ urls: 'stun:stun.l.google.com:19302' }];
        this.pc = null;
        this.sessionId = null;
        this.channels = {};
        this.rpcCallbacks = new Map();
        this.streamListeners = new Map();
        this.terminalListeners = [];
        this.statusListeners = [];
        this.rpcId = 1;
        this.isConnected = false;
    }

    onStatusChange(cb) {
        this.statusListeners.push(cb);
    }

    _notifyStatus(status, detail = null) {
        this.isConnected = status === 'connected';
        for (const cb of this.statusListeners) {
            try { cb(status, detail); } catch (e) { console.error('PyleWebRTC status listener error:', e); }
        }
    }

    async connect(token = null) {
        if (this.pc) {
            this.disconnect();
        }

        this._notifyStatus('connecting');

        const config = { iceServers: this.iceServers };
        this.pc = new RTCPeerConnection(config);

        // Pre-create standard channels so SDP negotiation includes SCTP media
        const channelLabels = ['control', 'stream', 'terminal'];
        for (const label of channelLabels) {
            const ch = this.pc.createDataChannel(label, { ordered: true });
            this._setupChannel(ch);
        }

        // Also handle incoming channels from host
        this.pc.ondatachannel = (event) => {
            this._setupChannel(event.channel);
        };

        this.pc.onicecandidate = async (event) => {
            if (event.candidate && this.sessionId) {
                try {
                    await fetch(`${this.apiBase}/api/webrtc/ice-candidate`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                        },
                        body: JSON.stringify({
                            session_id: this.sessionId,
                            candidate: event.candidate.toJSON ? event.candidate.toJSON() : event.candidate
                        })
                    });
                } catch (err) {
                    console.warn('Failed to send ICE candidate:', err);
                }
            }
        };

        this.pc.onconnectionstatechange = () => {
            const state = this.pc.connectionState;
            if (state === 'connected') {
                this._notifyStatus('connected');
            } else if (state === 'disconnected' || state === 'failed' || state === 'closed') {
                this._notifyStatus(state);
            }
        };

        // Create offer
        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);

        // Exchange offer via signaling endpoint
        const headers = {
            'Content-Type': 'application/json'
        };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const resp = await fetch(`${this.apiBase}/api/webrtc/offer`, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type
            })
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Offer failed' }));
            this._notifyStatus('failed', err.detail);
            this.disconnect();
            throw new Error(`WebRTC signaling error: ${err.detail || resp.statusText}`);
        }

        const answer = await resp.json();
        this.sessionId = answer.session_id;

        await this.pc.setRemoteDescription(new RTCSessionDescription({
            sdp: answer.sdp,
            type: answer.type
        }));

        return this.sessionId;
    }

    _setupChannel(channel) {
        const label = channel.label;
        this.channels[label] = channel;

        channel.onopen = () => {
            console.log(`[PyleWebRTC] Channel '${label}' opened`);
            if (label === 'control') {
                this.sendPing();
            }
        };

        channel.onclose = () => {
            console.log(`[PyleWebRTC] Channel '${label}' closed`);
            delete this.channels[label];
        };

        channel.onmessage = (event) => {
            this._handleChannelMessage(label, event.data);
        };
    }

    _handleChannelMessage(label, data) {
        if (label === 'control') {
            try {
                const msg = JSON.parse(data);
                if (msg.id && this.rpcCallbacks.has(msg.id)) {
                    const { resolve, reject } = this.rpcCallbacks.get(msg.id);
                    this.rpcCallbacks.delete(msg.id);
                    if (msg.error) {
                        reject(new Error(msg.error.message || 'RPC Error'));
                    } else {
                        resolve(msg.result);
                    }
                } else if (msg.method === 'welcome') {
                    console.log('[PyleWebRTC] Received welcome from host:', msg.params);
                }
            } catch (err) {
                console.warn('[PyleWebRTC] Invalid JSON on control channel:', err);
            }
        } else if (label === 'stream') {
            // Parse SSE-style formatting: "event: ...\ndata: ...\n\n"
            const lines = data.split('\n');
            let eventType = 'message';
            let payload = '';

            for (const line of lines) {
                if (line.startsWith('event:')) {
                    eventType = line.slice(6).trim();
                } else if (line.startsWith('data:')) {
                    payload += line.slice(5).trim();
                }
            }

            if (payload) {
                let parsed = payload;
                try { parsed = JSON.parse(payload); } catch (_) {}
                const listeners = this.streamListeners.get(eventType) || [];
                for (const cb of listeners) {
                    try { cb(parsed); } catch (e) { console.error('Stream listener error:', e); }
                }
            }
        } else if (label === 'terminal') {
            for (const cb of this.terminalListeners) {
                try { cb(data); } catch (e) { console.error('Terminal listener error:', e); }
            }
        }
    }

    sendRpc(method, params = {}) {
        return new Promise((resolve, reject) => {
            const channel = this.channels['control'];
            if (!channel || channel.readyState !== 'open') {
                return reject(new Error("WebRTC 'control' channel is not open"));
            }

            const id = this.rpcId++;
            this.rpcCallbacks.set(id, { resolve, reject });

            const payload = JSON.stringify({
                jsonrpc: '2.0',
                method,
                params,
                id
            });
            channel.send(payload);

            // 15s timeout
            setTimeout(() => {
                if (this.rpcCallbacks.has(id)) {
                    this.rpcCallbacks.delete(id);
                    reject(new Error(`RPC timeout for method '${method}'`));
                }
            }, 15000);
        });
    }

    sendPing() {
        return this.sendRpc('ping');
    }

    getAuthStatus() {
        return this.sendRpc('auth.status');
    }

    signPhylaxApproval(params) {
        return this.sendRpc('phylax.sign_approval', params);
    }

    onStream(eventType, callback) {
        if (!this.streamListeners.has(eventType)) {
            this.streamListeners.set(eventType, []);
        }
        this.streamListeners.get(eventType).push(callback);
    }

    sendTerminal(input) {
        const channel = this.channels['terminal'];
        if (channel && channel.readyState === 'open') {
            channel.send(input);
        }
    }

    onTerminal(callback) {
        this.terminalListeners.push(callback);
    }

    disconnect() {
        for (const label in this.channels) {
            try { this.channels[label].close(); } catch (_) {}
        }
        this.channels = {};
        if (this.pc) {
            try { this.pc.close(); } catch (_) {}
            this.pc = null;
        }
        this.sessionId = null;
        this.isConnected = false;
        this._notifyStatus('closed');
    }
}

// Export for ES modules or attach to window
if (typeof window !== 'undefined') {
    window.PyleWebRTCClient = PyleWebRTCClient;
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PyleWebRTCClient };
}
