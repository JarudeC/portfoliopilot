"use client";

import { useState } from 'react';
import Editor from '@monaco-editor/react';
import { Check, X, Edit3, Play, RotateCcw, Lock } from 'lucide-react';

interface CodeEditorProps {
  code: string;
  onApprove: (editedCode: string) => void;
  onReject: () => void;
  mode: 'forecast' | 'backtest';
  loading?: boolean;
}

export default function CodeEditor({ 
  code, 
  onApprove, 
  onReject, 
  mode,
  loading = false 
}: CodeEditorProps) {
  const [editedCode, setEditedCode] = useState(code);
  const [isEditing, setIsEditing] = useState(false);

  const handleApprove = () => {
    onApprove(editedCode);
  };

  const handleReset = () => {
    setEditedCode(code);
    setIsEditing(false);
  };

  const functionName = mode === 'forecast' ? 'generatePredictions' : 'calculateWeights';

  return (
    <div className="bg-[#0D1B2A] border border-[#4CC9F0]/30 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[#4CC9F0]/20 bg-[#14273F]">
        <div className="flex items-center gap-3">
          <Edit3 className="w-5 h-5 text-[#4CC9F0]" />
          <div>
            <h3 className="font-semibold text-white">Generated Code Review</h3>
            <p className="text-sm text-gray-400">
              Review and optionally edit the generated {functionName} function
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsEditing(!isEditing)}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              isEditing 
                ? 'bg-[#4CC9F0] text-[#0D1B2A] hover:bg-[#3A86FF]'
                : 'border border-[#4CC9F0] text-[#4CC9F0] hover:bg-[#4CC9F0]/10'
            }`}
            disabled={loading}
          >
            {isEditing ? 'View Mode' : 'Edit Mode'}
          </button>
        </div>
      </div>

      {/* Code Editor */}
      <div className="relative">
        <Editor
          height="400px"
          defaultLanguage="typescript"
          value={editedCode}
          onChange={(value) => setEditedCode(value || '')}
          theme="vs-dark"
          options={{
            readOnly: !isEditing,
            minimap: { enabled: false },
            fontSize: 14,
            lineNumbers: 'on',
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            automaticLayout: true,
            contextmenu: isEditing,
            selectOnLineNumbers: isEditing,
          }}
        />
        
        {/* Editing indicator */}
        {isEditing && (
          <div className="absolute top-2 right-2 bg-[#4CC9F0] text-[#0D1B2A] px-2 py-1 rounded text-xs font-medium">
            EDITING
          </div>
        )}
      </div>

      {/* Security Notice */}
      <div className="p-3 bg-[#14273F]/50 border-t border-[#4CC9F0]/20">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Lock className="w-4 h-4 text-[#4CC9F0]" />
          All code undergoes security validation before execution. 
          Network access, file operations, and potentially harmful patterns are blocked.
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between p-4 bg-[#14273F] border-t border-[#4CC9F0]/20">
        <div className="flex items-center gap-3">
          {isEditing && editedCode !== code && (
            <button
              onClick={handleReset}
              className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
              disabled={loading}
            >
              <RotateCcw className="w-4 h-4" />
              Reset to Original
            </button>
          )}
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={onReject}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-gray-400 hover:text-white border border-gray-600 hover:border-gray-500 rounded-lg transition-colors disabled:opacity-50"
          >
            <X className="w-4 h-4" />
            Cancel
          </button>
          
          <button
            onClick={handleApprove}
            disabled={loading || !editedCode.trim()}
            className="flex items-center gap-2 bg-[#4CC9F0] hover:bg-[#3A86FF] text-[#0D1B2A] font-semibold px-6 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-[#0D1B2A]/30 border-t-[#0D1B2A] rounded-full animate-spin" />
                Validating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Code
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}