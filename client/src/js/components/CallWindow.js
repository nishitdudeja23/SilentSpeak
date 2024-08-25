import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import classnames from 'classnames';
import { faPhone, faVideo } from '@fortawesome/free-solid-svg-icons';
import ActionButton from './ActionButton';

function CallWindow({ peerSrc, localSrc, config, mediaDevice, status, endCall }) {
  const peerVideo = useRef(null);
  const localVideo = useRef(null);
  const [video, setVideo] = useState(config.video);
  const [audio, setAudio] = useState(config.audio);
  const [captions, setCaptions] = useState('');
  const wsRef = useRef(null);

  useEffect(() => {
    if (peerVideo.current && peerSrc) peerVideo.current.srcObject = peerSrc;
    if (localVideo.current && localSrc) localVideo.current.srcObject = localSrc;
  }, [peerSrc, localSrc]);

  useEffect(() => {
    if (mediaDevice) {
      mediaDevice.toggle('Video', video);
      mediaDevice.toggle('Audio', audio);
    }
  }, [video, audio, mediaDevice]);

  useEffect(() => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws');

    wsRef.current.onmessage = (event) => {
      const recognizedText = event.data;
      setCaptions(recognizedText);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    const setupWebcam = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (localVideo.current) {
        localVideo.current.srcObject = stream;
        localVideo.current.play();
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      const interval = setInterval(() => {
        if (localVideo.current && localVideo.current.readyState === 4) {
          canvas.width = localVideo.current.videoWidth;
          canvas.height = localVideo.current.videoHeight;
          ctx.drawImage(localVideo.current, 0, 0, canvas.width, canvas.height);

          const imageDataURL = canvas.toDataURL('image/jpeg');
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ image: imageDataURL }));
          }
        }
      }, 33);

      return () => clearInterval(interval);
    };

    setupWebcam();
  }, []);

  const toggleMediaDevice = (deviceType) => {
    if (deviceType === 'Video') {
      setVideo(!video);
    }
    if (deviceType === 'Audio') {
      setAudio(!audio);
    }
    mediaDevice.toggle(deviceType);
  };

  return (
    <div className={classnames('call-window', status)}>
      <video id="peerVideo" ref={peerVideo} autoPlay />
      <video id="localVideo" ref={localVideo} autoPlay muted />
      <div className="video-control">
        <ActionButton
          key="btnVideo"
          icon={faVideo}
          disabled={!video}
          onClick={() => toggleMediaDevice('Video')}
        />
        <ActionButton
          key="btnAudio"
          icon={faPhone}
          disabled={!audio}
          onClick={() => toggleMediaDevice('Audio')}
        />
        <ActionButton
          className="hangup"
          icon={faPhone}
          onClick={() => endCall(true)}
        />
      </div>
      <div className="captions-container" style={{
        position: 'absolute',
        bottom: '10px',
        left: '50%',
        transform: 'translateX(-50%)',
        color: 'white',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        padding: '10px',
        borderRadius: '10px',
        textAlign: 'center',
        maxWidth: '90%',
        fontSize: '16px',
        fontFamily: 'Arial, sans-serif',
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)'
      }}>
        {captions}
      </div>
    </div>
  );
}

CallWindow.propTypes = {
  status: PropTypes.string.isRequired,
  localSrc: PropTypes.object,
  peerSrc: PropTypes.object,
  config: PropTypes.shape({
    audio: PropTypes.bool.isRequired,
    video: PropTypes.bool.isRequired
  }).isRequired,
  mediaDevice: PropTypes.object,
  endCall: PropTypes.func.isRequired
};

export default CallWindow;

